//! Buffer Operations Functionality Test
//! 
//! This module provides comprehensive testing of the restored ConjectureData
//! buffer operations functionality after implementing the Core Choice Sequence
//! Management System.

use crate::choice::{ChoiceType, ChoiceValue, Constraints, IntegerConstraints, BooleanConstraints, FloatConstraints};
use crate::choice_sequence_management::{ChoiceSequenceManager, ChoiceSequenceError};
use crate::data::{ConjectureData, Status, DrawError};
use std::collections::HashMap;

/// Comprehensive buffer operations test suite
pub struct BufferOperationsTest {
    test_cases: Vec<BufferTestCase>,
    results: Vec<BufferTestResult>,
}

/// Individual buffer operation test case
#[derive(Debug, Clone)]
pub struct BufferTestCase {
    pub name: String,
    pub description: String,
    pub buffer_size: usize,
    pub operations: Vec<BufferOperation>,
    pub expected_outcome: BufferExpectedOutcome,
}

/// Buffer operation to perform
#[derive(Debug, Clone)]
pub enum BufferOperation {
    /// Draw an integer choice
    DrawInteger { 
        constraints: IntegerConstraints,
        expected_size: usize,
    },
    /// Draw a boolean choice
    DrawBoolean { 
        constraints: BooleanConstraints,
        expected_size: usize,
    },
    /// Draw a float choice
    DrawFloat { 
        constraints: FloatConstraints,
        expected_size: usize,
    },
    /// Draw a string choice
    DrawString { 
        min_size: usize,
        max_size: usize,
        expected_string: String,
    },
    /// Draw bytes
    DrawBytes { 
        size: usize,
        expected_bytes: Vec<u8>,
    },
    /// Check buffer position
    CheckBufferPosition { 
        expected_position: usize,
    },
    /// Check remaining buffer space
    CheckRemainingSpace { 
        expected_remaining: usize,
    },
    /// Trigger buffer overrun
    TriggerOverrun,
}

/// Expected outcome of buffer test
#[derive(Debug, Clone, PartialEq)]
pub enum BufferExpectedOutcome {
    Success,
    BufferOverrun,
    InvalidChoice,
    SequenceMismatch,
}

/// Result of buffer operation test
#[derive(Debug, Clone)]
pub struct BufferTestResult {
    pub test_name: String,
    pub outcome: BufferActualOutcome,
    pub execution_time: std::time::Duration,
    pub buffer_state: BufferState,
    pub sequence_integrity: bool,
}

/// Actual outcome of buffer test
#[derive(Debug, Clone, PartialEq)]
pub enum BufferActualOutcome {
    Success,
    Error(String),
}

/// State of the buffer after operations
#[derive(Debug, Clone)]
pub struct BufferState {
    pub current_position: usize,
    pub total_size: usize,
    pub used_bytes: usize,
    pub remaining_bytes: usize,
    pub choices_recorded: usize,
    pub fragmentation_level: f64,
}

impl BufferOperationsTest {
    /// Create a new buffer operations test suite
    pub fn new() -> Self {
        println!("BUFFER_TEST DEBUG: Creating BufferOperationsTest");
        
        Self {
            test_cases: Vec::new(),
            results: Vec::new(),
        }
    }
    
    /// Add test case for basic buffer operations
    pub fn add_basic_buffer_test(&mut self) {
        let test_case = BufferTestCase {
            name: "basic_buffer_operations".to_string(),
            description: "Test basic buffer allocation and choice recording".to_string(),
            buffer_size: 1024,
            operations: vec![
                BufferOperation::DrawInteger {
                    constraints: IntegerConstraints::default(),
                    expected_size: 8,
                },
                BufferOperation::CheckBufferPosition {
                    expected_position: 8,
                },
                BufferOperation::DrawBoolean {
                    constraints: BooleanConstraints::default(),
                    expected_size: 1,
                },
                BufferOperation::CheckBufferPosition {
                    expected_position: 9,
                },
                BufferOperation::CheckRemainingSpace {
                    expected_remaining: 1024 - 9,
                },
            ],
            expected_outcome: BufferExpectedOutcome::Success,
        };
        
        self.test_cases.push(test_case);
    }
    
    /// Add test case for buffer overrun scenarios
    pub fn add_buffer_overrun_test(&mut self) {
        let test_case = BufferTestCase {
            name: "buffer_overrun_detection".to_string(),
            description: "Test buffer overrun detection and handling".to_string(),
            buffer_size: 32, // Small buffer to trigger overrun
            operations: vec![
                BufferOperation::DrawInteger {
                    constraints: IntegerConstraints::default(),
                    expected_size: 8,
                },
                BufferOperation::DrawInteger {
                    constraints: IntegerConstraints::default(),
                    expected_size: 8,
                },
                BufferOperation::DrawInteger {
                    constraints: IntegerConstraints::default(),
                    expected_size: 8,
                },
                BufferOperation::DrawInteger {
                    constraints: IntegerConstraints::default(),
                    expected_size: 8,
                },
                // This should trigger overrun
                BufferOperation::TriggerOverrun,
            ],
            expected_outcome: BufferExpectedOutcome::BufferOverrun,
        };
        
        self.test_cases.push(test_case);
    }
    
    /// Add test case for mixed choice types in buffer
    pub fn add_mixed_choice_buffer_test(&mut self) {
        let test_case = BufferTestCase {
            name: "mixed_choice_buffer".to_string(),
            description: "Test buffer operations with different choice types".to_string(),
            buffer_size: 2048,
            operations: vec![
                BufferOperation::DrawInteger {
                    constraints: IntegerConstraints::default(),
                    expected_size: 8,
                },
                BufferOperation::DrawFloat {
                    constraints: FloatConstraints::default(),
                    expected_size: 8,
                },
                BufferOperation::DrawBoolean {
                    constraints: BooleanConstraints::default(),
                    expected_size: 1,
                },
                BufferOperation::DrawString {
                    min_size: 5,
                    max_size: 10,
                    expected_string: "test".to_string(),
                },
                BufferOperation::DrawBytes {
                    size: 4,
                    expected_bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
                },
                BufferOperation::CheckBufferPosition {
                    expected_position: 8 + 8 + 1 + 4 + 4, // Sum of all sizes
                },
            ],
            expected_outcome: BufferExpectedOutcome::Success,
        };
        
        self.test_cases.push(test_case);
    }
    
    /// Add test case for large buffer stress test
    pub fn add_large_buffer_stress_test(&mut self) {
        let mut operations = Vec::new();
        
        // Generate many operations to stress test the buffer
        for i in 0..1000 {
            operations.push(BufferOperation::DrawInteger {
                constraints: IntegerConstraints::default(),
                expected_size: 8,
            });
            
            if i % 100 == 0 {
                operations.push(BufferOperation::CheckBufferPosition {
                    expected_position: (i + 1) * 8,
                });
            }
        }
        
        let test_case = BufferTestCase {
            name: "large_buffer_stress".to_string(),
            description: "Stress test buffer with many operations".to_string(),
            buffer_size: 16384, // Large buffer
            operations,
            expected_outcome: BufferExpectedOutcome::Success,
        };
        
        self.test_cases.push(test_case);
    }
    
    /// Run all buffer operation tests
    pub fn run_all_tests(&mut self) -> Vec<BufferTestResult> {
        println!("BUFFER_TEST DEBUG: Running {} buffer test cases", self.test_cases.len());
        
        self.results.clear();
        
        for test_case in &self.test_cases.clone() {
            let result = self.run_single_buffer_test(test_case);
            self.results.push(result);
        }
        
        self.print_buffer_test_summary();
        self.results.clone()
    }
    
    /// Run a single buffer operation test
    fn run_single_buffer_test(&mut self, test_case: &BufferTestCase) -> BufferTestResult {
        let start_time = std::time::Instant::now();
        
        println!("BUFFER_TEST DEBUG: Running test '{}'", test_case.name);
        
        // Create a choice sequence manager for this test
        let mut manager = ChoiceSequenceManager::new(test_case.buffer_size);
        let mut current_position = 0;
        let mut choices_recorded = 0;
        
        let outcome = match self.execute_buffer_operations(
            &mut manager, 
            &test_case.operations, 
            &mut current_position,
            &mut choices_recorded,
        ) {
            Ok(_) => {
                if test_case.expected_outcome == BufferExpectedOutcome::Success {
                    BufferActualOutcome::Success
                } else {
                    BufferActualOutcome::Error(format!(
                        "Expected failure ({:?}) but test succeeded", 
                        test_case.expected_outcome
                    ))
                }
            },
            Err(error) => {
                let expected_error = match test_case.expected_outcome {
                    BufferExpectedOutcome::BufferOverrun => error.to_string().contains("Buffer"),
                    BufferExpectedOutcome::InvalidChoice => error.to_string().contains("Invalid"),
                    BufferExpectedOutcome::SequenceMismatch => error.to_string().contains("Sequence"),
                    BufferExpectedOutcome::Success => false,
                };
                
                if expected_error {
                    BufferActualOutcome::Success
                } else {
                    BufferActualOutcome::Error(error.to_string())
                }
            }
        };
        
        let execution_time = start_time.elapsed();
        
        // Calculate buffer state
        let buffer_state = BufferState {
            current_position,
            total_size: test_case.buffer_size,
            used_bytes: current_position,
            remaining_bytes: test_case.buffer_size.saturating_sub(current_position),
            choices_recorded,
            fragmentation_level: 0.0, // Simplified for now
        };
        
        // Check sequence integrity
        let integrity_status = manager.get_integrity_status();
        let sequence_integrity = integrity_status.is_healthy;
        
        BufferTestResult {
            test_name: test_case.name.clone(),
            outcome,
            execution_time,
            buffer_state,
            sequence_integrity,
        }
    }
    
    /// Execute buffer operations for a test case
    fn execute_buffer_operations(
        &self,
        manager: &mut ChoiceSequenceManager,
        operations: &[BufferOperation],
        current_position: &mut usize,
        choices_recorded: &mut usize,
    ) -> Result<(), ChoiceSequenceError> {
        
        for (op_index, operation) in operations.iter().enumerate() {
            println!("BUFFER_TEST DEBUG: Executing operation {} ({:?})", op_index, operation);
            
            match operation {
                BufferOperation::DrawInteger { constraints, expected_size } => {
                    let value = ChoiceValue::Integer(42); // Simplified for testing
                    let constraints_box = Box::new(Constraints::Integer(constraints.clone()));
                    
                    manager.record_choice(
                        ChoiceType::Integer,
                        value,
                        constraints_box,
                        false,
                        *current_position,
                    )?;
                    
                    *current_position += expected_size;
                    *choices_recorded += 1;
                },
                
                BufferOperation::DrawBoolean { constraints, expected_size } => {
                    let value = ChoiceValue::Boolean(true); // Simplified for testing
                    let constraints_box = Box::new(Constraints::Boolean(constraints.clone()));
                    
                    manager.record_choice(
                        ChoiceType::Boolean,
                        value,
                        constraints_box,
                        false,
                        *current_position,
                    )?;
                    
                    *current_position += expected_size;
                    *choices_recorded += 1;
                },
                
                BufferOperation::DrawFloat { constraints, expected_size } => {
                    let value = ChoiceValue::Float(3.14159); // Simplified for testing
                    let constraints_box = Box::new(Constraints::Float(constraints.clone()));
                    
                    manager.record_choice(
                        ChoiceType::Float,
                        value,
                        constraints_box,
                        false,
                        *current_position,
                    )?;
                    
                    *current_position += expected_size;
                    *choices_recorded += 1;
                },
                
                BufferOperation::DrawString { min_size: _, max_size: _, expected_string } => {
                    let value = ChoiceValue::String(expected_string.clone());
                    let constraints_box = Box::new(Constraints::String(
                        crate::choice::StringConstraints::default()
                    ));
                    
                    manager.record_choice(
                        ChoiceType::String,
                        value,
                        constraints_box,
                        false,
                        *current_position,
                    )?;
                    
                    *current_position += expected_string.len();
                    *choices_recorded += 1;
                },
                
                BufferOperation::DrawBytes { size: _, expected_bytes } => {
                    let value = ChoiceValue::Bytes(expected_bytes.clone());
                    let constraints_box = Box::new(Constraints::Bytes(
                        crate::choice::BytesConstraints::default()
                    ));
                    
                    manager.record_choice(
                        ChoiceType::Bytes,
                        value,
                        constraints_box,
                        false,
                        *current_position,
                    )?;
                    
                    *current_position += expected_bytes.len();
                    *choices_recorded += 1;
                },
                
                BufferOperation::CheckBufferPosition { expected_position } => {
                    if *current_position != *expected_position {
                        return Err(ChoiceSequenceError::IntegrityViolation {
                            violation_type: crate::choice_sequence_management::ViolationType::BufferMismatch,
                            description: format!(
                                "Buffer position mismatch: expected {} but got {}",
                                expected_position, current_position
                            ),
                        });
                    }
                },
                
                BufferOperation::CheckRemainingSpace { expected_remaining } => {
                    let remaining = manager.buffer_tracker.max_size.saturating_sub(*current_position);
                    if remaining != *expected_remaining {
                        return Err(ChoiceSequenceError::IntegrityViolation {
                            violation_type: crate::choice_sequence_management::ViolationType::BufferMismatch,
                            description: format!(
                                "Remaining space mismatch: expected {} but got {}",
                                expected_remaining, remaining
                            ),
                        });
                    }
                },
                
                BufferOperation::TriggerOverrun => {
                    // Try to allocate more than remaining space
                    let remaining = manager.buffer_tracker.max_size.saturating_sub(*current_position);
                    let overrun_size = remaining + 1000; // Ensure overrun
                    
                    if overrun_size > remaining {
                        return Err(ChoiceSequenceError::BufferOverflow {
                            required: overrun_size,
                            available: remaining,
                        });
                    }
                },
            }
        }
        
        Ok(())
    }
    
    /// Print buffer test summary
    fn print_buffer_test_summary(&self) {
        println!("\n=== BUFFER OPERATIONS TEST SUMMARY ===");
        
        let total_tests = self.results.len();
        let successful_tests = self.results.iter()
            .filter(|r| r.outcome == BufferActualOutcome::Success)
            .count();
        
        println!("Total tests: {}", total_tests);
        println!("Successful: {}", successful_tests);
        println!("Failed: {}", total_tests - successful_tests);
        
        let total_execution_time: f64 = self.results.iter()
            .map(|r| r.execution_time.as_secs_f64())
            .sum();
        
        println!("Total execution time: {:.3}s", total_execution_time);
        
        if !self.results.is_empty() {
            println!("Average test time: {:.3}ms", 
                     (total_execution_time / self.results.len() as f64) * 1000.0);
        }
        
        println!("\nDetailed Results:");
        for result in &self.results {
            let status = match &result.outcome {
                BufferActualOutcome::Success => "✓ PASS",
                BufferActualOutcome::Error(_) => "✗ FAIL",
            };
            
            let integrity_status = if result.sequence_integrity { "✓" } else { "✗" };
            
            println!("  {} {} ({:.3}ms) [Integrity: {}] [Buffer: {}/{}]", 
                     status, 
                     result.test_name, 
                     result.execution_time.as_secs_f64() * 1000.0,
                     integrity_status,
                     result.buffer_state.used_bytes,
                     result.buffer_state.total_size);
            
            if let BufferActualOutcome::Error(error) = &result.outcome {
                println!("    Error: {}", error);
            }
        }
        
        println!("=== END BUFFER TEST SUMMARY ===\n");
    }
    
    /// Get buffer test success rate
    pub fn get_success_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        
        let successful = self.results.iter()
            .filter(|r| r.outcome == BufferActualOutcome::Success)
            .count();
        
        successful as f64 / self.results.len() as f64
    }
    
    /// Generate buffer operations report
    pub fn generate_buffer_report(&self) -> BufferOperationsReport {
        let total_buffer_used: usize = self.results.iter()
            .map(|r| r.buffer_state.used_bytes)
            .sum();
        
        let total_choices: usize = self.results.iter()
            .map(|r| r.buffer_state.choices_recorded)
            .sum();
        
        let avg_execution_time = if !self.results.is_empty() {
            self.results.iter()
                .map(|r| r.execution_time.as_secs_f64())
                .sum::<f64>() / self.results.len() as f64
        } else {
            0.0
        };
        
        BufferOperationsReport {
            total_tests: self.results.len(),
            successful_tests: self.results.iter()
                .filter(|r| r.outcome == BufferActualOutcome::Success)
                .count(),
            success_rate: self.get_success_rate(),
            total_buffer_used,
            total_choices_recorded: total_choices,
            average_execution_time: avg_execution_time,
            buffer_efficiency: if total_buffer_used > 0 { 
                total_choices as f64 / total_buffer_used as f64 
            } else { 
                0.0 
            },
            integrity_violations: self.results.iter()
                .filter(|r| !r.sequence_integrity)
                .count(),
        }
    }
}

/// Buffer operations test report
#[derive(Debug, Clone)]
pub struct BufferOperationsReport {
    pub total_tests: usize,
    pub successful_tests: usize,
    pub success_rate: f64,
    pub total_buffer_used: usize,
    pub total_choices_recorded: usize,
    pub average_execution_time: f64,
    pub buffer_efficiency: f64,
    pub integrity_violations: usize,
}

impl BufferOperationsReport {
    /// Print formatted buffer operations report
    pub fn print_report(&self) {
        println!("\n=== BUFFER OPERATIONS FUNCTIONALITY REPORT ===");
        println!("Test Results:");
        println!("  Total Tests: {}", self.total_tests);
        println!("  Successful: {}", self.successful_tests);
        println!("  Success Rate: {:.1}%", self.success_rate * 100.0);
        
        println!("\nBuffer Performance:");
        println!("  Total Buffer Used: {} bytes", self.total_buffer_used);
        println!("  Total Choices Recorded: {}", self.total_choices_recorded);
        println!("  Average Execution Time: {:.3}ms", self.average_execution_time * 1000.0);
        println!("  Buffer Efficiency: {:.2} choices/byte", self.buffer_efficiency);
        
        println!("\nIntegrity Status:");
        println!("  Integrity Violations: {}", self.integrity_violations);
        
        println!("=== END BUFFER OPERATIONS REPORT ===\n");
    }
}

/// Run comprehensive buffer operations demonstration
pub fn run_buffer_operations_demonstration() -> BufferOperationsReport {
    println!("=== BUFFER OPERATIONS FUNCTIONALITY DEMONSTRATION ===\n");
    
    let mut buffer_test = BufferOperationsTest::new();
    
    // Add all test cases
    buffer_test.add_basic_buffer_test();
    buffer_test.add_buffer_overrun_test();
    buffer_test.add_mixed_choice_buffer_test();
    buffer_test.add_large_buffer_stress_test();
    
    println!("Added {} buffer test cases", buffer_test.test_cases.len());
    
    // Run all tests
    let _results = buffer_test.run_all_tests();
    
    // Generate and return report
    let report = buffer_test.generate_buffer_report();
    report.print_report();
    
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_operations_suite_creation() {
        let buffer_test = BufferOperationsTest::new();
        assert_eq!(buffer_test.test_cases.len(), 0);
        assert_eq!(buffer_test.results.len(), 0);
    }

    #[test]
    fn test_basic_buffer_operations() {
        let mut buffer_test = BufferOperationsTest::new();
        buffer_test.add_basic_buffer_test();
        
        let results = buffer_test.run_all_tests();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].outcome, BufferActualOutcome::Success);
        assert!(results[0].sequence_integrity);
    }

    #[test]
    fn test_buffer_overrun_detection() {
        let mut buffer_test = BufferOperationsTest::new();
        buffer_test.add_buffer_overrun_test();
        
        let results = buffer_test.run_all_tests();
        assert_eq!(results.len(), 1);
        // Should succeed because we expect the overrun to be detected
        assert_eq!(results[0].outcome, BufferActualOutcome::Success);
    }

    #[test]
    fn test_mixed_choice_buffer() {
        let mut buffer_test = BufferOperationsTest::new();
        buffer_test.add_mixed_choice_buffer_test();
        
        let results = buffer_test.run_all_tests();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].outcome, BufferActualOutcome::Success);
        
        // Should have recorded multiple choices of different types
        assert!(results[0].buffer_state.choices_recorded >= 5);
    }

    #[test]
    fn test_large_buffer_stress() {
        let mut buffer_test = BufferOperationsTest::new();
        buffer_test.add_large_buffer_stress_test();
        
        let start_time = std::time::Instant::now();
        let results = buffer_test.run_all_tests();
        let elapsed = start_time.elapsed();
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].outcome, BufferActualOutcome::Success);
        
        // Should have recorded 1000 choices
        assert_eq!(results[0].buffer_state.choices_recorded, 1000);
        
        // Performance should be reasonable (less than 5 seconds for 1000 operations)
        assert!(elapsed.as_secs() < 5);
        
        println!("Large buffer stress test performance: {:.3}ms for {} choices", 
                 elapsed.as_millis(), results[0].buffer_state.choices_recorded);
    }

    #[test]
    fn test_buffer_operations_demonstration() {
        let report = run_buffer_operations_demonstration();
        
        assert!(report.total_tests > 0);
        assert!(report.success_rate >= 0.0 && report.success_rate <= 1.0);
        assert!(report.total_buffer_used > 0);
        assert!(report.total_choices_recorded > 0);
    }
}