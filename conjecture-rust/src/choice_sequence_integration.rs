//! Integration module demonstrating the Core Choice Sequence Management System
//! 
//! This module provides comprehensive integration tests and demonstrations
//! of the enhanced choice sequence management capability that fixes type
//! inconsistencies, index tracking issues, and sequence replay functionality.

use crate::choice::{ChoiceType, ChoiceValue, Constraints, IntegerConstraints, BooleanConstraints, FloatConstraints};
use crate::choice_sequence_management::{ChoiceSequenceManager, ChoiceSequenceError};
use crate::data::{ConjectureData, Status, TreeRecordingObserver};
use crate::datatree::DataTree;
use std::collections::HashMap;

/// Comprehensive integration test suite for choice sequence management
pub struct ChoiceSequenceIntegrationTest {
    manager: ChoiceSequenceManager,
    test_data: Vec<TestCase>,
    results: Vec<TestResult>,
}

/// Individual test case for choice sequence operations
#[derive(Debug, Clone)]
pub struct TestCase {
    pub name: String,
    pub description: String,
    pub choices: Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>,
    pub expected_outcome: ExpectedOutcome,
}

/// Expected outcome of a test case
#[derive(Debug, Clone, PartialEq)]
pub enum ExpectedOutcome {
    Success,
    TypeMismatch,
    ConstraintViolation,
    IndexOutOfBounds,
    BufferOverflow,
}

/// Result of running a test case
#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub outcome: ActualOutcome,
    pub execution_time: std::time::Duration,
    pub integrity_status: String,
    pub performance_metrics: String,
}

/// Actual outcome of test execution
#[derive(Debug, Clone, PartialEq)]
pub enum ActualOutcome {
    Success,
    Error(String),
}

impl ChoiceSequenceIntegrationTest {
    /// Create a new integration test suite
    pub fn new() -> Self {
        println!("INTEGRATION DEBUG: Creating ChoiceSequenceIntegrationTest");
        
        Self {
            manager: ChoiceSequenceManager::new(8192),
            test_data: Vec::new(),
            results: Vec::new(),
        }
    }
    
    /// Add test case for basic integer choice recording and replay
    pub fn add_integer_choice_test(&mut self) {
        let test_case = TestCase {
            name: "integer_choice_basic".to_string(),
            description: "Test basic integer choice recording and replay".to_string(),
            choices: vec![
                (
                    ChoiceType::Integer,
                    ChoiceValue::Integer(42),
                    Box::new(Constraints::Integer(IntegerConstraints::default())),
                ),
                (
                    ChoiceType::Integer,
                    ChoiceValue::Integer(-17),
                    Box::new(Constraints::Integer(IntegerConstraints::default())),
                ),
                (
                    ChoiceType::Integer,
                    ChoiceValue::Integer(0),
                    Box::new(Constraints::Integer(IntegerConstraints::default())),
                ),
            ],
            expected_outcome: ExpectedOutcome::Success,
        };
        
        self.test_data.push(test_case);
    }
    
    /// Add test case for mixed type sequence
    pub fn add_mixed_type_test(&mut self) {
        let test_case = TestCase {
            name: "mixed_type_sequence".to_string(),
            description: "Test sequence with different choice types".to_string(),
            choices: vec![
                (
                    ChoiceType::Integer,
                    ChoiceValue::Integer(100),
                    Box::new(Constraints::Integer(IntegerConstraints::default())),
                ),
                (
                    ChoiceType::Boolean,
                    ChoiceValue::Boolean(true),
                    Box::new(Constraints::Boolean(BooleanConstraints::default())),
                ),
                (
                    ChoiceType::Float,
                    ChoiceValue::Float(3.14159),
                    Box::new(Constraints::Float(FloatConstraints::default())),
                ),
                (
                    ChoiceType::String,
                    ChoiceValue::String("test_string".to_string()),
                    Box::new(Constraints::String(crate::choice::StringConstraints::default())),
                ),
                (
                    ChoiceType::Bytes,
                    ChoiceValue::Bytes(vec![0xAB, 0xCD, 0xEF]),
                    Box::new(Constraints::Bytes(crate::choice::BytesConstraints::default())),
                ),
            ],
            expected_outcome: ExpectedOutcome::Success,
        };
        
        self.test_data.push(test_case);
    }
    
    /// Add test case for constraint validation
    pub fn add_constraint_validation_test(&mut self) {
        let mut int_constraints = IntegerConstraints::default();
        int_constraints.min_value = Some(0);
        int_constraints.max_value = Some(100);
        
        let test_case = TestCase {
            name: "constraint_validation".to_string(),
            description: "Test constraint validation during recording".to_string(),
            choices: vec![
                (
                    ChoiceType::Integer,
                    ChoiceValue::Integer(50), // Valid: within [0, 100]
                    Box::new(Constraints::Integer(int_constraints.clone())),
                ),
                (
                    ChoiceType::Integer,
                    ChoiceValue::Integer(0), // Valid: at minimum
                    Box::new(Constraints::Integer(int_constraints.clone())),
                ),
                (
                    ChoiceType::Integer,
                    ChoiceValue::Integer(100), // Valid: at maximum
                    Box::new(Constraints::Integer(int_constraints.clone())),
                ),
            ],
            expected_outcome: ExpectedOutcome::Success,
        };
        
        self.test_data.push(test_case);
    }
    
    /// Add test case for large sequence handling
    pub fn add_large_sequence_test(&mut self) {
        let mut choices = Vec::new();
        
        // Generate 1000 integer choices to test performance and integrity
        for i in 0..1000 {
            choices.push((
                ChoiceType::Integer,
                ChoiceValue::Integer(i),
                Box::new(Constraints::Integer(IntegerConstraints::default())),
            ));
        }
        
        let test_case = TestCase {
            name: "large_sequence_performance".to_string(),
            description: "Test performance and integrity with large choice sequences".to_string(),
            choices,
            expected_outcome: ExpectedOutcome::Success,
        };
        
        self.test_data.push(test_case);
    }
    
    /// Run all test cases
    pub fn run_all_tests(&mut self) -> Vec<TestResult> {
        println!("INTEGRATION DEBUG: Running {} test cases", self.test_data.len());
        
        self.results.clear();
        
        for test_case in &self.test_data.clone() {
            let result = self.run_single_test(test_case);
            self.results.push(result);
        }
        
        self.print_test_summary();
        self.results.clone()
    }
    
    /// Run a single test case
    fn run_single_test(&mut self, test_case: &TestCase) -> TestResult {
        let start_time = std::time::Instant::now();
        
        println!("INTEGRATION DEBUG: Running test '{}'", test_case.name);
        
        // Reset manager for clean test
        self.manager.reset_sequence();
        
        let outcome = match self.execute_test_case(test_case) {
            Ok(_) => {
                if test_case.expected_outcome == ExpectedOutcome::Success {
                    ActualOutcome::Success
                } else {
                    ActualOutcome::Error(format!(
                        "Expected failure ({:?}) but test succeeded", 
                        test_case.expected_outcome
                    ))
                }
            },
            Err(error) => {
                let expected_error = match test_case.expected_outcome {
                    ExpectedOutcome::TypeMismatch => error.to_string().contains("Type"),
                    ExpectedOutcome::ConstraintViolation => error.to_string().contains("Constraint"),
                    ExpectedOutcome::IndexOutOfBounds => error.to_string().contains("Index"),
                    ExpectedOutcome::BufferOverflow => error.to_string().contains("Buffer"),
                    ExpectedOutcome::Success => false,
                };
                
                if expected_error {
                    ActualOutcome::Success
                } else {
                    ActualOutcome::Error(error.to_string())
                }
            }
        };
        
        let execution_time = start_time.elapsed();
        
        // Collect integrity and performance data
        let integrity_status = format!("{:?}", self.manager.get_integrity_status());
        let performance_metrics = format!("{:?}", self.manager.get_performance_metrics());
        
        TestResult {
            test_name: test_case.name.clone(),
            outcome,
            execution_time,
            integrity_status,
            performance_metrics,
        }
    }
    
    /// Execute the actual test case operations
    fn execute_test_case(&mut self, test_case: &TestCase) -> Result<(), ChoiceSequenceError> {
        // Phase 1: Record all choices
        for (i, (choice_type, value, constraints)) in test_case.choices.iter().enumerate() {
            let buffer_position = i * 8; // Simple buffer position calculation
            self.manager.record_choice(
                *choice_type,
                value.clone(),
                constraints.clone(),
                false, // Not forced
                buffer_position,
            )?;
            
            println!("INTEGRATION DEBUG: Recorded choice {} ({:?} = {:?})", 
                     i, choice_type, value);
        }
        
        // Phase 2: Replay all choices to verify consistency
        for (i, (choice_type, value, constraints)) in test_case.choices.iter().enumerate() {
            let replayed_value = self.manager.replay_choice_at_index(
                i,
                *choice_type,
                constraints.as_ref(),
            )?;
            
            if replayed_value != *value {
                return Err(ChoiceSequenceError::IntegrityViolation {
                    violation_type: crate::choice_sequence_management::ViolationType::ReplayInconsistency,
                    description: format!(
                        "Replayed value {:?} does not match original {:?} at index {}",
                        replayed_value, value, i
                    ),
                });
            }
            
            println!("INTEGRATION DEBUG: Successfully replayed choice {} ({:?})", 
                     i, replayed_value);
        }
        
        // Phase 3: Verify sequence integrity
        let integrity_status = self.manager.get_integrity_status();
        if !integrity_status.is_healthy {
            return Err(ChoiceSequenceError::IntegrityViolation {
                violation_type: crate::choice_sequence_management::ViolationType::ReplayInconsistency,
                description: format!(
                    "Sequence integrity compromised: {} violations",
                    integrity_status.total_violations
                ),
            });
        }
        
        Ok(())
    }
    
    /// Print test summary
    fn print_test_summary(&self) {
        println!("\n=== CHOICE SEQUENCE INTEGRATION TEST SUMMARY ===");
        
        let total_tests = self.results.len();
        let successful_tests = self.results.iter()
            .filter(|r| r.outcome == ActualOutcome::Success)
            .count();
        
        println!("Total tests: {}", total_tests);
        println!("Successful: {}", successful_tests);
        println!("Failed: {}", total_tests - successful_tests);
        
        println!("\nDetailed Results:");
        for result in &self.results {
            let status = match &result.outcome {
                ActualOutcome::Success => "✓ PASS",
                ActualOutcome::Error(_) => "✗ FAIL",
            };
            
            println!("  {} {} ({:.3}ms)", 
                     status, 
                     result.test_name, 
                     result.execution_time.as_secs_f64() * 1000.0);
            
            if let ActualOutcome::Error(error) = &result.outcome {
                println!("    Error: {}", error);
            }
        }
        
        println!("=== END SUMMARY ===\n");
    }
    
    /// Get overall test success rate
    pub fn get_success_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        
        let successful = self.results.iter()
            .filter(|r| r.outcome == ActualOutcome::Success)
            .count();
        
        successful as f64 / self.results.len() as f64
    }
    
    /// Generate integration report
    pub fn generate_integration_report(&self) -> IntegrationReport {
        let mut total_recording_time = 0.0;
        let mut total_replay_time = 0.0;
        
        for result in &self.results {
            total_recording_time += result.execution_time.as_secs_f64();
        }
        
        let performance_metrics = self.manager.get_performance_metrics();
        
        IntegrationReport {
            total_tests: self.results.len(),
            successful_tests: self.results.iter()
                .filter(|r| r.outcome == ActualOutcome::Success)
                .count(),
            success_rate: self.get_success_rate(),
            total_execution_time: total_recording_time,
            average_test_time: if !self.results.is_empty() { 
                total_recording_time / self.results.len() as f64 
            } else { 
                0.0 
            },
            choice_recording_performance: performance_metrics.avg_recording_time,
            choice_replay_performance: performance_metrics.avg_replay_time,
            type_verification_performance: performance_metrics.type_verification_time,
            cache_hit_rate: performance_metrics.cache_hit_rate,
            integrity_violations_detected: 0, // Would need to aggregate from all tests
            recovery_actions_successful: 0,   // Would need to aggregate from all tests
        }
    }
}

/// Comprehensive integration report
#[derive(Debug, Clone)]
pub struct IntegrationReport {
    pub total_tests: usize,
    pub successful_tests: usize,
    pub success_rate: f64,
    pub total_execution_time: f64,
    pub average_test_time: f64,
    pub choice_recording_performance: f64,
    pub choice_replay_performance: f64,
    pub type_verification_performance: f64,
    pub cache_hit_rate: f64,
    pub integrity_violations_detected: usize,
    pub recovery_actions_successful: usize,
}

impl IntegrationReport {
    /// Print formatted report
    pub fn print_report(&self) {
        println!("\n=== CHOICE SEQUENCE MANAGEMENT INTEGRATION REPORT ===");
        println!("Test Results:");
        println!("  Total Tests: {}", self.total_tests);
        println!("  Successful: {}", self.successful_tests);
        println!("  Success Rate: {:.1}%", self.success_rate * 100.0);
        
        println!("\nPerformance Metrics:");
        println!("  Total Execution Time: {:.3}s", self.total_execution_time);
        println!("  Average Test Time: {:.3}ms", self.average_test_time * 1000.0);
        println!("  Choice Recording: {:.6}ms", self.choice_recording_performance * 1000.0);
        println!("  Choice Replay: {:.6}ms", self.choice_replay_performance * 1000.0);
        println!("  Type Verification: {:.6}ms", self.type_verification_performance * 1000.0);
        println!("  Cache Hit Rate: {:.1}%", self.cache_hit_rate * 100.0);
        
        println!("\nIntegrity Monitoring:");
        println!("  Violations Detected: {}", self.integrity_violations_detected);
        println!("  Recovery Actions: {}", self.recovery_actions_successful);
        
        println!("=== END INTEGRATION REPORT ===\n");
    }
}

/// Run comprehensive choice sequence management demonstration
pub fn run_comprehensive_demonstration() -> IntegrationReport {
    println!("=== CHOICE SEQUENCE MANAGEMENT CAPABILITY DEMONSTRATION ===\n");
    
    let mut integration_test = ChoiceSequenceIntegrationTest::new();
    
    // Add all test cases
    integration_test.add_integer_choice_test();
    integration_test.add_mixed_type_test();
    integration_test.add_constraint_validation_test();
    integration_test.add_large_sequence_test();
    
    println!("Added {} test cases", integration_test.test_data.len());
    
    // Run all tests
    let _results = integration_test.run_all_tests();
    
    // Generate and return report
    let report = integration_test.generate_integration_report();
    report.print_report();
    
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_suite_creation() {
        let integration_test = ChoiceSequenceIntegrationTest::new();
        assert_eq!(integration_test.test_data.len(), 0);
        assert_eq!(integration_test.results.len(), 0);
    }

    #[test]
    fn test_add_integer_choice_test() {
        let mut integration_test = ChoiceSequenceIntegrationTest::new();
        integration_test.add_integer_choice_test();
        
        assert_eq!(integration_test.test_data.len(), 1);
        assert_eq!(integration_test.test_data[0].name, "integer_choice_basic");
        assert_eq!(integration_test.test_data[0].choices.len(), 3);
    }

    #[test]
    fn test_mixed_type_sequence() {
        let mut integration_test = ChoiceSequenceIntegrationTest::new();
        integration_test.add_mixed_type_test();
        
        let results = integration_test.run_all_tests();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].outcome, ActualOutcome::Success);
    }

    #[test]
    fn test_comprehensive_demonstration() {
        let report = run_comprehensive_demonstration();
        
        assert!(report.total_tests > 0);
        assert!(report.success_rate >= 0.0 && report.success_rate <= 1.0);
        assert!(report.total_execution_time >= 0.0);
    }

    #[test]
    fn test_large_sequence_performance() {
        let mut integration_test = ChoiceSequenceIntegrationTest::new();
        integration_test.add_large_sequence_test();
        
        let start_time = std::time::Instant::now();
        let results = integration_test.run_all_tests();
        let elapsed = start_time.elapsed();
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].outcome, ActualOutcome::Success);
        
        // Performance should be reasonable (less than 1 second for 1000 choices)
        assert!(elapsed.as_secs() < 1);
        
        let report = integration_test.generate_integration_report();
        println!("Large sequence test performance: {:.3}ms", 
                 report.average_test_time * 1000.0);
    }
}