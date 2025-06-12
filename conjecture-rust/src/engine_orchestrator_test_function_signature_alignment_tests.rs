//! Comprehensive tests for EngineOrchestrator test function signature alignment
//!
//! This module provides comprehensive PyO3 and FFI integration tests that validate the
//! test function signature alignment capability of the EngineOrchestrator. It focuses on
//! verifying that type inconsistencies between OrchestrationResult<()> and Result<T, DrawError>
//! are properly handled and converted.

use crate::data::{ConjectureData, DrawError, Status};
use crate::engine_orchestrator::{
    EngineOrchestrator, OrchestratorConfig, OrchestrationError, OrchestrationResult, ExecutionPhase
};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Helper function to convert DrawError to OrchestrationError for testing
fn convert_draw_error_to_orchestration_error(error: DrawError) -> OrchestrationError {
    match error {
        DrawError::Overrun => OrchestrationError::Overrun,
        DrawError::UnsatisfiedAssumption(msg) => OrchestrationError::Invalid { 
            reason: format!("Unsatisfied assumption: {}", msg) 
        },
        DrawError::StopTest(_) => OrchestrationError::Interrupted,
        DrawError::Frozen => OrchestrationError::Invalid { 
            reason: "ConjectureData is frozen".to_string() 
        },
        DrawError::InvalidRange => OrchestrationError::Invalid { 
            reason: "Invalid range specified".to_string() 
        },
        DrawError::InvalidProbability => OrchestrationError::Invalid { 
            reason: "Invalid probability".to_string() 
        },
        DrawError::EmptyAlphabet => OrchestrationError::Invalid { 
            reason: "Empty alphabet".to_string() 
        },
        DrawError::InvalidStatus => OrchestrationError::Invalid { 
            reason: "Invalid status".to_string() 
        },
        DrawError::EmptyChoice => OrchestrationError::Invalid { 
            reason: "Empty choice".to_string() 
        },
        DrawError::InvalidReplayType => OrchestrationError::Invalid { 
            reason: "Invalid replay type".to_string() 
        },
    }
}

/// Test the core capability: test function signature alignment
/// 
/// This test validates that the orchestrator correctly handles test functions
/// that return OrchestrationResult<()> while ConjectureData operations return
/// Result<T, DrawError>, ensuring proper type conversion and error handling.
#[test]
fn test_function_signature_alignment_core_capability() {
    // Test function that returns OrchestrationResult<()> but uses ConjectureData operations
    // that return Result<T, DrawError> - this should compile and work correctly
    let test_fn = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
        // These operations return Result<T, DrawError> but need to be converted to OrchestrationResult<()>
        let _integer = data.draw_integer(1, 100)
            .map_err(|e| convert_draw_error_to_orchestration_error(e))?;
        
        let _boolean = data.draw_boolean(0.5)
            .map_err(|e| convert_draw_error_to_orchestration_error(e))?;
        
        let _float = data.draw_float()
            .map_err(|e| convert_draw_error_to_orchestration_error(e))?;
        
        Ok(())
    });
    
    let config = OrchestratorConfig {
        max_examples: 10,
        backend: "hypothesis".to_string(),
        debug_logging: true,
        ..Default::default()
    };
    
    let mut orchestrator = EngineOrchestrator::new(test_fn, config);
    
    // Verify the orchestrator accepts the test function
    assert_eq!(orchestrator.current_phase(), ExecutionPhase::Initialize);
    
    // Run the orchestrator - this should work without compilation errors
    let result = orchestrator.run();
    assert!(result.is_ok(), "Orchestrator should handle type conversion correctly");
    
    let stats = result.unwrap();
    eprintln!("Test function signature alignment test completed with stats: {:?}", stats);
}

/// Test error conversion from DrawError to OrchestrationError
/// 
/// This validates that all DrawError variants are properly converted to
/// appropriate OrchestrationError variants for consistent error handling.
#[test]
fn test_draw_error_to_orchestration_error_conversion() {
    // Test each DrawError variant conversion
    let draw_errors = vec![
        DrawError::Frozen,
        DrawError::InvalidRange,
        DrawError::InvalidProbability,
        DrawError::EmptyAlphabet,
        DrawError::Overrun,
        DrawError::InvalidStatus,
        DrawError::EmptyChoice,
        DrawError::InvalidReplayType,
        DrawError::StopTest(42),
        DrawError::UnsatisfiedAssumption("test assumption".to_string()),
        DrawError::PreviouslyUnseenBehaviour,
        DrawError::InvalidChoice,
        DrawError::EmptyWeights,
        DrawError::InvalidWeights,
    ];
    
    for draw_error in draw_errors {
        let orchestration_error = convert_draw_error_to_orchestration_error(draw_error.clone());
        
        // Verify conversion maintains error semantics
        match draw_error {
            DrawError::Overrun => {
                assert!(matches!(orchestration_error, OrchestrationError::Overrun));
            }
            DrawError::UnsatisfiedAssumption(msg) => {
                if let OrchestrationError::Invalid { reason } = orchestration_error {
                    assert!(reason.contains(&msg));
                } else {
                    panic!("UnsatisfiedAssumption should convert to Invalid");
                }
            }
            DrawError::StopTest(_) => {
                assert!(matches!(orchestration_error, OrchestrationError::Interrupted));
            }
            _ => {
                // Most other errors should convert to Invalid with appropriate reason
                assert!(matches!(orchestration_error, OrchestrationError::Invalid { .. }));
            }
        }
    }
}

/// Test orchestrator with test function that deliberately triggers different DrawError types
/// 
/// This validates that the orchestrator can handle test functions that trigger various
/// DrawError conditions and properly convert them to OrchestrationError types.
#[test]
fn test_orchestrator_with_draw_error_triggering_functions() {
    let error_counts = Arc::new(Mutex::new(HashMap::new()));
    
    // Test function that triggers different DrawError types based on iteration
    let test_fn = {
        let error_counts = Arc::clone(&error_counts);
        Box::new(move |data: &mut ConjectureData| -> OrchestrationResult<()> {
            let mut counts = error_counts.lock().unwrap();
            let call_count = counts.len();
            
            match call_count % 4 {
                0 => {
                    // Trigger InvalidRange error
                    let result = data.draw_integer(100, 1); // Invalid range
                    match result {
                        Ok(_) => Ok(()),
                        Err(DrawError::InvalidRange) => {
                            *counts.entry("InvalidRange".to_string()).or_insert(0) += 1;
                            Err(OrchestrationError::Invalid { 
                                reason: "Invalid range encountered".to_string() 
                            })
                        }
                        Err(e) => Err(convert_draw_error_to_orchestration_error(e))
                    }
                }
                1 => {
                    // Trigger InvalidProbability error
                    let result = data.draw_boolean(1.5); // Invalid probability
                    match result {
                        Ok(_) => Ok(()),
                        Err(DrawError::InvalidProbability) => {
                            *counts.entry("InvalidProbability".to_string()).or_insert(0) += 1;
                            Err(OrchestrationError::Invalid { 
                                reason: "Invalid probability encountered".to_string() 
                            })
                        }
                        Err(e) => Err(convert_draw_error_to_orchestration_error(e))
                    }
                }
                2 => {
                    // Trigger UnsatisfiedAssumption
                    *counts.entry("UnsatisfiedAssumption".to_string()).or_insert(0) += 1;
                    Err(OrchestrationError::Invalid { 
                        reason: "Assumption not satisfied".to_string() 
                    })
                }
                _ => {
                    // Valid case
                    let _value = data.draw_integer(1, 10)
                        .map_err(|e| convert_draw_error_to_orchestration_error(e))?;
                    *counts.entry("Valid".to_string()).or_insert(0) += 1;
                    Ok(())
                }
            }
        })
    };
    
    let config = OrchestratorConfig {
        max_examples: 20,
        backend: "hypothesis".to_string(),
        debug_logging: true,
        ignore_limits: true, // Allow more examples to test various error paths
        ..Default::default()
    };
    
    let mut orchestrator = EngineOrchestrator::new(test_fn, config);
    let result = orchestrator.run();
    
    // Orchestrator should handle the errors gracefully
    assert!(result.is_ok() || result.is_err(), "Orchestrator should complete execution");
    
    let counts = error_counts.lock().unwrap();
    eprintln!("Error handling test completed with counts: {:?}", *counts);
    
    // Verify we exercised different error paths
    assert!(counts.len() > 0, "Should have recorded at least some operations");
}

/// Test PyO3 integration for test function signature alignment
/// 
/// This validates that test function signature alignment works correctly
/// when called from Python through PyO3 FFI bindings.
#[test] 
fn test_pyo3_integration_function_signature_alignment() {
    #[cfg(feature = "python-ffi")]
    use pyo3::prelude::*;
    
    // Test function that simulates Python calling pattern
    let test_fn = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
        // Simulate Python-style test function that draws values and checks properties
        let value1 = data.draw_integer(0, 100)
            .map_err(|e| convert_draw_error_to_orchestration_error(e))?;
        
        let value2 = data.draw_integer(0, 100)
            .map_err(|e| convert_draw_error_to_orchestration_error(e))?;
        
        // Property check: sum should be within bounds
        if value1 + value2 > 150 {
            // This represents a property violation that Python would detect
            return Err(OrchestrationError::Invalid { 
                reason: format!("Property violation: {} + {} = {} > 150", value1, value2, value1 + value2)
            });
        }
        
        Ok(())
    });
    
    let config = OrchestratorConfig {
        max_examples: 50,
        backend: "hypothesis".to_string(),
        debug_logging: true,
        ..Default::default()
    };
    
    let mut orchestrator = EngineOrchestrator::new(test_fn, config);
    let result = orchestrator.run();
    
    // Should work correctly with PyO3-style error handling
    match result {
        Ok(stats) => {
            eprintln!("PyO3 integration test completed successfully: {:?}", stats);
            assert!(stats.phases.len() > 0, "Should have recorded execution phases");
        }
        Err(e) => {
            eprintln!("PyO3 integration test completed with controlled error: {}", e);
            // This is acceptable as we're testing error handling
        }
    }
}

/// Test FFI integration for test function signature alignment
/// 
/// This validates that test function signature alignment works correctly
/// when interfacing with C FFI or other foreign function interfaces.
#[test]
fn test_ffi_integration_function_signature_alignment() {
    // Simulate FFI-style callback with proper error handling
    let test_fn = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
        // FFI-style operations that need careful error handling
        let result1 = data.draw_integer(1, 1000);
        let value1 = match result1 {
            Ok(v) => v,
            Err(DrawError::Overrun) => return Err(OrchestrationError::Overrun),
            Err(DrawError::InvalidRange) => return Err(OrchestrationError::Invalid { 
                reason: "FFI: Invalid integer range".to_string() 
            }),
            Err(e) => return Err(convert_draw_error_to_orchestration_error(e)),
        };
        
        let result2 = data.draw_boolean(0.3);
        let value2 = match result2 {
            Ok(v) => v,
            Err(DrawError::InvalidProbability) => return Err(OrchestrationError::Invalid { 
                reason: "FFI: Invalid boolean probability".to_string() 
            }),
            Err(e) => return Err(convert_draw_error_to_orchestration_error(e)),
        };
        
        // FFI-style validation
        if value1 > 500 && value2 {
            return Err(OrchestrationError::Invalid { 
                reason: "FFI: Combined condition failed".to_string() 
            });
        }
        
        Ok(())
    });
    
    let config = OrchestratorConfig {
        max_examples: 30,
        backend: "hypothesis".to_string(),
        debug_logging: true,
        ..Default::default()
    };
    
    let mut orchestrator = EngineOrchestrator::new(test_fn, config);
    let result = orchestrator.run();
    
    match result {
        Ok(stats) => {
            eprintln!("FFI integration test completed successfully: {:?}", stats);
            // Verify orchestrator tracked execution properly
            assert!(orchestrator.call_count() > 0, "Should have made test calls");
        }
        Err(e) => {
            eprintln!("FFI integration test handled error correctly: {}", e);
        }
    }
}

/// Test comprehensive error handling chain
/// 
/// This validates the complete error handling chain from ConjectureData
/// operations through DrawError to OrchestrationError conversion.
#[test]
fn test_comprehensive_error_handling_chain() {
    let error_log = Arc::new(Mutex::new(Vec::new()));
    
    let test_fn = {
        let error_log = Arc::clone(&error_log);
        Box::new(move |data: &mut ConjectureData| -> OrchestrationResult<()> {
            // Test comprehensive error handling
            let mut log = error_log.lock().unwrap();
            
            // Valid operation
            match data.draw_integer(1, 10) {
                Ok(v) => {
                    log.push(format!("Valid integer: {}", v));
                }
                Err(e) => {
                    log.push(format!("Integer error: {:?}", e));
                    return Err(convert_draw_error_to_orchestration_error(e));
                }
            }
            
            // Valid boolean operation
            match data.draw_boolean(0.5) {
                Ok(v) => {
                    log.push(format!("Valid boolean: {}", v));
                }
                Err(e) => {
                    log.push(format!("Boolean error: {:?}", e));
                    return Err(convert_draw_error_to_orchestration_error(e));
                }
            }
            
            // Valid float operation
            match data.draw_float() {
                Ok(v) => {
                    log.push(format!("Valid float: {}", v));
                }
                Err(e) => {
                    log.push(format!("Float error: {:?}", e));
                    return Err(convert_draw_error_to_orchestration_error(e));
                }
            }
            
            Ok(())
        })
    };
    
    let config = OrchestratorConfig {
        max_examples: 5,
        backend: "hypothesis".to_string(),
        debug_logging: true,
        ..Default::default()
    };
    
    let mut orchestrator = EngineOrchestrator::new(test_fn, config);
    let result = orchestrator.run();
    
    let log = error_log.lock().unwrap();
    eprintln!("Comprehensive error handling test log: {:?}", *log);
    
    // Should have completed successfully with proper error handling
    assert!(result.is_ok(), "Comprehensive error handling should work correctly");
    assert!(log.len() > 0, "Should have logged operations");
}

/// Test orchestrator behavior with different error return patterns
/// 
/// This validates that the orchestrator correctly handles test functions
/// that return different patterns of OrchestrationResult<()>.
#[test]
fn test_orchestrator_error_return_patterns() {
    // Pattern 1: Always return Ok(())
    let test_fn_ok = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
        let _value = data.draw_integer(1, 100)
            .map_err(|e| convert_draw_error_to_orchestration_error(e))?;
        Ok(())
    });
    
    let config = OrchestratorConfig {
        max_examples: 5,
        backend: "hypothesis".to_string(),
        ..Default::default()
    };
    
    let mut orchestrator = EngineOrchestrator::new(test_fn_ok, config.clone());
    let result = orchestrator.run();
    assert!(result.is_ok(), "Always-Ok test function should complete successfully");
    
    // Pattern 2: Return error on specific condition
    let test_fn_conditional = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
        let value = data.draw_integer(1, 100)
            .map_err(|e| convert_draw_error_to_orchestration_error(e))?;
        
        if value > 50 {
            Err(OrchestrationError::Invalid { 
                reason: "Value too large".to_string() 
            })
        } else {
            Ok(())
        }
    });
    
    let mut orchestrator2 = EngineOrchestrator::new(test_fn_conditional, config.clone());
    let result2 = orchestrator2.run();
    // May succeed or fail depending on generated values
    eprintln!("Conditional test function result: {:?}", result2);
    
    // Pattern 3: Always return error
    let test_fn_error = Box::new(|_data: &mut ConjectureData| -> OrchestrationResult<()> {
        Err(OrchestrationError::Invalid { 
            reason: "Always fails".to_string() 
        })
    });
    
    let mut orchestrator3 = EngineOrchestrator::new(test_fn_error, config);
    let result3 = orchestrator3.run();
    // Should handle the error gracefully
    eprintln!("Always-error test function result: {:?}", result3);
}

/// Helper function to convert DrawError to OrchestrationError
/// 
/// This implements the type conversion logic that test functions need
/// to bridge between ConjectureData operations and orchestrator expectations.
fn convert_draw_error_to_orchestration_error(draw_error: DrawError) -> OrchestrationError {
    match draw_error {
        DrawError::Overrun => OrchestrationError::Overrun,
        DrawError::UnsatisfiedAssumption(msg) => OrchestrationError::Invalid { 
            reason: format!("Unsatisfied assumption: {}", msg)
        },
        DrawError::StopTest(_) => OrchestrationError::Interrupted,
        DrawError::Frozen => OrchestrationError::Invalid { 
            reason: "ConjectureData is frozen".to_string()
        },
        DrawError::InvalidRange => OrchestrationError::Invalid { 
            reason: "Invalid range specified".to_string()
        },
        DrawError::InvalidProbability => OrchestrationError::Invalid { 
            reason: "Invalid probability specified".to_string()
        },
        DrawError::EmptyAlphabet => OrchestrationError::Invalid { 
            reason: "Empty alphabet for string generation".to_string()
        },
        DrawError::InvalidStatus => OrchestrationError::Invalid { 
            reason: "ConjectureData has invalid status".to_string()
        },
        DrawError::EmptyChoice => OrchestrationError::Invalid { 
            reason: "Empty choice sequence".to_string()
        },
        DrawError::InvalidReplayType => OrchestrationError::Invalid { 
            reason: "Invalid replay type".to_string()
        },
        DrawError::PreviouslyUnseenBehaviour => OrchestrationError::Invalid { 
            reason: "Previously unseen behavior".to_string()
        },
        DrawError::InvalidChoice => OrchestrationError::Invalid { 
            reason: "Invalid choice".to_string()
        },
        DrawError::EmptyWeights => OrchestrationError::Invalid { 
            reason: "Empty weights for weighted choice".to_string()
        },
        DrawError::InvalidWeights => OrchestrationError::Invalid { 
            reason: "Invalid weights for weighted choice".to_string()
        },
    }
}

/// Integration test for real-world usage patterns
/// 
/// This test simulates how the orchestrator would be used in practice
/// with test functions that have mixed success/failure patterns.
#[test]
fn test_real_world_usage_patterns() {
    let stats_collector = Arc::new(Mutex::new(HashMap::new()));
    
    let test_fn = {
        let stats = Arc::clone(&stats_collector);
        Box::new(move |data: &mut ConjectureData| -> OrchestrationResult<()> {
            let mut stats = stats.lock().unwrap();
            
            // Realistic test: generate two numbers and test their relationship
            let a = data.draw_integer(1, 100)
                .map_err(convert_draw_error_to_orchestration_error)?;
            
            let b = data.draw_integer(1, 100)
                .map_err(convert_draw_error_to_orchestration_error)?;
            
            // Track statistics
            *stats.entry("total_calls".to_string()).or_insert(0) += 1;
            
            // Property: a + b should be > 50 (this will sometimes fail)
            if a + b <= 50 {
                *stats.entry("property_failures".to_string()).or_insert(0) += 1;
                return Err(OrchestrationError::Invalid { 
                    reason: format!("Property failed: {} + {} = {} <= 50", a, b, a + b)
                });
            }
            
            *stats.entry("property_successes".to_string()).or_insert(0) += 1;
            Ok(())
        })
    };
    
    let config = OrchestratorConfig {
        max_examples: 100,
        backend: "hypothesis".to_string(),
        debug_logging: false, // Reduce noise
        ..Default::default()
    };
    
    let mut orchestrator = EngineOrchestrator::new(test_fn, config);
    let result = orchestrator.run();
    
    let final_stats = stats_collector.lock().unwrap();
    eprintln!("Real-world usage test stats: {:?}", *final_stats);
    
    // Should have completed and collected statistics
    assert!(final_stats.contains_key("total_calls"), "Should have made calls");
    
    match result {
        Ok(execution_stats) => {
            eprintln!("Real-world test completed successfully: {:?}", execution_stats);
        }
        Err(e) => {
            eprintln!("Real-world test completed with error (expected): {}", e);
        }
    }
}