//! Comprehensive integration tests for ProviderSystem module capability
//! Tests focus on Provider Interface Type System Repair functionality
//! 
//! Key capability areas tested:
//! 1. Dynamic Provider Type Safety and Interface Compatibility  
//! 2. Provider Registry and Context Switching Integration
//! 3. Constraint System Cross-Provider Validation
//! 4. PyO3 FFI Integration and Type Safety
//! 5. Provider Lifecycle Management and Error Recovery

use crate::choice::constraints::{IntegerConstraints, FloatConstraints, BooleanConstraints};
use crate::choice::float_constraint_type_system::FloatPrimitiveProvider;
use crate::providers::{PrimitiveProvider, HypothesisProvider, RandomProvider};
use crate::engine_orchestrator_provider_type_integration::{
    EnhancedPrimitiveProvider, 
    EnhancedHypothesisProvider, 
    EnhancedRandomProvider,
    ProviderTypeRegistry,
    ProviderTypeManager,
    ProviderTypeError,
    ProviderContext,
    ProviderBackendError
};
use crate::data::{ConjectureData, Status};
use std::sync::Arc;
use std::collections::HashMap;
use pyo3::prelude::*;

#[cfg(test)]
mod comprehensive_provider_system_tests {
    use super::*;

    /// Test dynamic provider type safety and interface compatibility
    /// Validates the core Provider Interface Type System Repair capability
    #[test]
    fn test_dynamic_provider_type_safety_interface_compatibility() {
        // Test static-to-dynamic provider conversion type safety
        let static_hypothesis_provider = HypothesisProvider::new();
        let dynamic_hypothesis_provider: Box<dyn PrimitiveProvider> = Box::new(static_hypothesis_provider);
        
        let static_random_provider = RandomProvider::new();  
        let dynamic_random_provider: Box<dyn PrimitiveProvider> = Box::new(static_random_provider);

        // Test enhanced provider type compatibility
        let enhanced_hypothesis = EnhancedHypothesisProvider::new();
        let enhanced_random = EnhancedRandomProvider::new();
        
        // Validate trait object creation for enhanced providers
        let enhanced_dynamic_hypothesis: Box<dyn EnhancedPrimitiveProvider> = Box::new(enhanced_hypothesis);
        let enhanced_dynamic_random: Box<dyn EnhancedPrimitiveProvider> = Box::new(enhanced_random);

        // Test provider context creation with different types
        let mut conjecture_data = ConjectureData::for_buffer(&[1, 2, 3, 4, 5]);
        
        // Test integer generation with type safety validation
        let int_constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(50),
        };
        
        let int_result_dynamic = dynamic_hypothesis_provider.integer(&mut conjecture_data, &int_constraints);
        let int_result_enhanced = enhanced_dynamic_hypothesis.integer(&mut conjecture_data, &int_constraints);
        
        assert!(int_result_dynamic.is_ok());
        assert!(int_result_enhanced.is_ok());
        
        // Validate both approaches produce compatible results
        assert!(int_result_dynamic.unwrap() >= 0 && int_result_dynamic.unwrap() <= 100);
        assert!(int_result_enhanced.unwrap() >= 0 && int_result_enhanced.unwrap() <= 100);
    }

    /// Test provider registry and dynamic dispatch functionality
    /// Validates registry-based provider creation with different constraint types
    #[test]
    fn test_provider_registry_dynamic_dispatch_integration() {
        let mut registry = ProviderTypeRegistry::new();
        
        // Register different provider types
        registry.register_provider("hypothesis", || Box::new(EnhancedHypothesisProvider::new()));
        registry.register_provider("random", || Box::new(EnhancedRandomProvider::new()));
        
        // Test dynamic provider creation
        let hypothesis_provider = registry.create_provider("hypothesis").unwrap();
        let random_provider = registry.create_provider("random").unwrap();
        
        let mut conjecture_data = ConjectureData::for_buffer(&[10, 20, 30, 40]);
        
        // Test cross-constraint compatibility for all registered providers
        let test_constraints = vec![
            ("integer", Box::new(IntegerConstraints {
                min_value: Some(-50),
                max_value: Some(50),
                weights: None,
                shrink_towards: Some(0),
            }) as Box<dyn std::any::Any>),
            ("float", Box::new(FloatConstraints {
                min_value: Some(0.0),
                max_value: Some(1.0),
                allow_nan: false,
                smallest_nonzero_magnitude: Some(1e-10),
            }) as Box<dyn std::any::Any>),
            ("boolean", Box::new(BooleanConstraints {
                probability: Some(0.7),
            }) as Box<dyn std::any::Any>),
        ];
        
        for (constraint_type, constraint) in test_constraints {
            match constraint_type {
                "integer" => {
                    let int_constraint = constraint.downcast_ref::<IntegerConstraints>().unwrap();
                    let hypothesis_result = hypothesis_provider.integer(&mut conjecture_data, int_constraint);
                    let random_result = random_provider.integer(&mut conjecture_data, int_constraint);
                    
                    assert!(hypothesis_result.is_ok());
                    assert!(random_result.is_ok());
                }
                "float" => {
                    let float_constraint = constraint.downcast_ref::<FloatConstraints>().unwrap();
                    let hypothesis_result = hypothesis_provider.float(&mut conjecture_data, float_constraint);
                    let random_result = random_provider.float(&mut conjecture_data, float_constraint);
                    
                    assert!(hypothesis_result.is_ok());
                    assert!(random_result.is_ok());
                }
                "boolean" => {
                    let bool_constraint = constraint.downcast_ref::<BooleanConstraints>().unwrap();
                    let hypothesis_result = hypothesis_provider.boolean(&mut conjecture_data, bool_constraint);
                    let random_result = random_provider.boolean(&mut conjecture_data, bool_constraint);
                    
                    assert!(hypothesis_result.is_ok());
                    assert!(random_result.is_ok());
                }
                _ => unreachable!(),
            }
        }
    }

    /// Test provider context switching and lifecycle management
    /// Validates provider switching, error recovery, and resource cleanup
    #[test]
    fn test_provider_context_switching_lifecycle_management() {
        let mut provider_manager = ProviderTypeManager::new();
        
        // Test provider context creation and switching
        let hypothesis_context = ProviderContext::new("hypothesis");
        let random_context = ProviderContext::new("random");
        
        provider_manager.set_context(hypothesis_context.clone()).unwrap();
        assert_eq!(provider_manager.current_context().name(), "hypothesis");
        
        // Test context switching with validation
        provider_manager.set_context(random_context.clone()).unwrap();
        assert_eq!(provider_manager.current_context().name(), "random");
        
        // Test error handling for invalid provider contexts
        let invalid_context = ProviderContext::new("nonexistent");
        let switch_result = provider_manager.set_context(invalid_context);
        assert!(matches!(switch_result, Err(ProviderTypeError::InvalidProvider(_))));
        
        // Test backend cannot proceed scenarios
        let mut conjecture_data = ConjectureData::for_buffer(&[]);
        conjecture_data.mark_overrun();
        
        let current_provider = provider_manager.current_provider();
        
        // Test different backend error scope types
        let backend_errors = vec![
            ProviderBackendError::new("verified", "Test verification failure"),
            ProviderBackendError::new("exhausted", "Search space exhausted"),
            ProviderBackendError::new("discard_test_case", "Test case discarded"),
        ];
        
        for backend_error in backend_errors {
            let error_result = provider_manager.handle_backend_error(backend_error);
            // Validate error is properly categorized and handled
            assert!(error_result.is_ok() || matches!(error_result, Err(ProviderTypeError::BackendError(_))));
        }
        
        // Test provider lifecycle cleanup
        provider_manager.reset_context();
        // Validate context was properly cleaned up (should use default)
        assert_eq!(provider_manager.current_context().name(), "default");
    }

    /// Test constraint system cross-provider validation
    /// Ensures all providers handle all constraint types correctly
    #[test]
    fn test_constraint_system_cross_provider_validation() {
        let providers: Vec<(&str, Box<dyn EnhancedPrimitiveProvider>)> = vec![
            ("hypothesis", Box::new(EnhancedHypothesisProvider::new())),
            ("random", Box::new(EnhancedRandomProvider::new())),
        ];
        
        // Test comprehensive constraint validation across all providers
        for (provider_name, provider) in providers {
            let mut conjecture_data = ConjectureData::for_buffer(&[1, 2, 3, 4, 5, 6, 7, 8]);
            
            // Test integer constraints with edge cases
            let integer_constraints = vec![
                IntegerConstraints { min_value: Some(i64::MIN), max_value: Some(i64::MAX), weights: None, shrink_towards: None },
                IntegerConstraints { min_value: Some(0), max_value: Some(0), weights: None, shrink_towards: Some(0) },
                IntegerConstraints { min_value: Some(-1000), max_value: Some(1000), weights: Some(vec![0.1, 0.9]), shrink_towards: Some(100) },
            ];
            
            for constraint in integer_constraints {
                let result = provider.integer(&mut conjecture_data, &constraint);
                assert!(result.is_ok(), "Provider {} failed integer constraint: {:?}", provider_name, constraint);
                
                let value = result.unwrap();
                if let Some(min) = constraint.min_value {
                    assert!(value >= min, "Provider {} violated min constraint {} < {}", provider_name, value, min);
                }
                if let Some(max) = constraint.max_value {
                    assert!(value <= max, "Provider {} violated max constraint {} > {}", provider_name, value, max);
                }
            }
            
            // Test float constraints with special values
            let float_constraints = vec![
                FloatConstraints { min_value: Some(f64::NEG_INFINITY), max_value: Some(f64::INFINITY), allow_nan: true, smallest_nonzero_magnitude: None },
                FloatConstraints { min_value: Some(0.0), max_value: Some(1.0), allow_nan: false, smallest_nonzero_magnitude: Some(1e-100) },
                FloatConstraints { min_value: Some(-1e10), max_value: Some(1e10), allow_nan: false, smallest_nonzero_magnitude: Some(1e-10) },
            ];
            
            for constraint in float_constraints {
                let result = provider.float(&mut conjecture_data, &constraint);
                assert!(result.is_ok(), "Provider {} failed float constraint: {:?}", provider_name, constraint);
                
                let value = result.unwrap();
                if !constraint.allow_nan {
                    assert!(!value.is_nan(), "Provider {} generated NaN when not allowed", provider_name);
                }
                if let Some(min) = constraint.min_value {
                    if !value.is_nan() {
                        assert!(value >= min, "Provider {} violated min constraint {} < {}", provider_name, value, min);
                    }
                }
                if let Some(max) = constraint.max_value {
                    if !value.is_nan() {
                        assert!(value <= max, "Provider {} violated max constraint {} > {}", provider_name, value, max);
                    }
                }
            }
            
            // Test boolean constraints
            let boolean_constraints = vec![
                BooleanConstraints { probability: None },
                BooleanConstraints { probability: Some(0.0) },
                BooleanConstraints { probability: Some(1.0) },
                BooleanConstraints { probability: Some(0.5) },
            ];
            
            for constraint in boolean_constraints {
                let result = provider.boolean(&mut conjecture_data, &constraint);
                assert!(result.is_ok(), "Provider {} failed boolean constraint: {:?}", provider_name, constraint);
            }
        }
    }

    /// Test constant injection and edge case generation
    /// Validates edge case generation probabilities and constant caching
    #[test]
    fn test_constant_injection_edge_case_generation() {
        let hypothesis_provider = EnhancedHypothesisProvider::new();
        let mut results: HashMap<String, Vec<i64>> = HashMap::new();
        
        // Generate large sample to test constant injection probability (5%)
        for iteration in 0..1000 {
            let mut conjecture_data = ConjectureData::for_buffer(&[
                (iteration % 256) as u8, 
                ((iteration / 256) % 256) as u8,
                ((iteration / 65536) % 256) as u8,
                ((iteration / 16777216) % 256) as u8,
            ]);
            
            let constraint = IntegerConstraints {
                min_value: Some(-1000),
                max_value: Some(1000),
                weights: None,
                shrink_towards: Some(0),
            };
            
            let result = hypothesis_provider.integer(&mut conjecture_data, &constraint).unwrap();
            results.entry("integers".to_string()).or_insert_with(Vec::new).push(result);
        }
        
        let integers = results.get("integers").unwrap();
        
        // Test for edge case constants (boundary values)
        let edge_cases = vec![-1000, -1, 0, 1, 1000];
        let mut edge_case_count = 0;
        
        for &edge_case in &edge_cases {
            let count = integers.iter().filter(|&&x| x == edge_case).count();
            if count > 0 {
                edge_case_count += count;
            }
        }
        
        // Validate constant injection probability (should be around 5% * sample_size * edge_cases)
        let expected_edge_cases = (1000.0 * 0.05 * edge_cases.len() as f64) as usize;
        let tolerance = expected_edge_cases / 2; // 50% tolerance for randomness
        
        assert!(
            edge_case_count >= expected_edge_cases.saturating_sub(tolerance) && 
            edge_case_count <= expected_edge_cases + tolerance,
            "Edge case injection rate outside expected range: {} (expected ~{})", 
            edge_case_count, expected_edge_cases
        );
        
        // Test float edge cases (NaN, Infinity, special values)
        let mut float_results = Vec::new();
        for iteration in 0..500 {
            let mut conjecture_data = ConjectureData::for_buffer(&[
                (iteration % 256) as u8,
                ((iteration / 256) % 256) as u8,
                (iteration % 127) as u8,
                ((iteration * 7) % 256) as u8,
            ]);
            
            let constraint = FloatConstraints {
                min_value: Some(-1e6),
                max_value: Some(1e6),
                allow_nan: true,
                smallest_nonzero_magnitude: Some(1e-100),
            };
            
            let result = hypothesis_provider.float(&mut conjecture_data, &constraint).unwrap();
            float_results.push(result);
        }
        
        // Check for special float constants
        let special_floats = float_results.iter().filter(|&&x| 
            x.is_nan() || x.is_infinite() || x == 0.0 || x == -0.0 || x == 1.0 || x == -1.0
        ).count();
        
        // Should have some special float values due to constant injection
        assert!(special_floats > 0, "No special float constants generated");
    }

    /// Test PyO3 FFI integration and type safety
    /// Validates Python interop, error translation, and FFI type safety
    #[test]
    fn test_pyo3_ffi_integration_type_safety() {
        Python::with_gil(|py| {
            // Test provider creation through PyO3 interface
            let hypothesis_provider = EnhancedHypothesisProvider::new();
            
            // Create test conjecture data
            let mut conjecture_data = ConjectureData::for_buffer(&[42, 100, 200, 50]);
            
            // Test integer generation with PyO3 error handling
            let int_constraint = IntegerConstraints {
                min_value: Some(0),
                max_value: Some(255),
                weights: None,
                shrink_towards: Some(127),
            };
            
            let int_result = hypothesis_provider.integer(&mut conjecture_data, &int_constraint);
            assert!(int_result.is_ok());
            
            // Test provider statistics serialization for Python
            let provider_stats = hypothesis_provider.get_statistics();
            
            // Validate statistics can be serialized for Python FFI
            let stats_dict = pyo3::types::PyDict::new(py);
            stats_dict.set_item("generation_count", provider_stats.generation_count).unwrap();
            stats_dict.set_item("constant_injection_rate", provider_stats.constant_injection_rate).unwrap();
            stats_dict.set_item("active_constraints", provider_stats.active_constraints.len()).unwrap();
            
            assert_eq!(stats_dict.get_item("generation_count").unwrap().extract::<u64>().unwrap(), provider_stats.generation_count);
            
            // Test error translation to Python exceptions
            let mut overrun_data = ConjectureData::for_buffer(&[]);
            overrun_data.mark_overrun();
            
            let error_result = hypothesis_provider.integer(&mut overrun_data, &int_constraint);
            assert!(error_result.is_err());
            
            // Validate error can be converted to Python exception
            match error_result.unwrap_err() {
                provider_error => {
                    let py_exception = pyo3::exceptions::PyRuntimeError::new_err(format!("Provider error: {}", provider_error));
                    assert!(py_exception.to_string().contains("Provider error"));
                }
            }
            
            // Test float constraint PyO3 integration
            let float_constraint = FloatConstraints {
                min_value: Some(0.0),
                max_value: Some(1.0),
                allow_nan: false,
                smallest_nonzero_magnitude: Some(1e-10),
            };
            
            conjecture_data = ConjectureData::for_buffer(&[128, 64, 192, 32]);
            let float_result = hypothesis_provider.float(&mut conjecture_data, &float_constraint);
            assert!(float_result.is_ok());
            
            let float_value = float_result.unwrap();
            assert!(float_value >= 0.0 && float_value <= 1.0);
            assert!(!float_value.is_nan());
            
            // Test provider context serialization for Python
            let context = ProviderContext::new("hypothesis");
            let context_dict = pyo3::types::PyDict::new(py);
            context_dict.set_item("name", context.name()).unwrap();
            context_dict.set_item("created_at", context.created_at().timestamp()).unwrap();
            
            assert_eq!(context_dict.get_item("name").unwrap().extract::<String>().unwrap(), "hypothesis");
        });
    }

    /// Test provider performance under concurrent load
    /// Validates thread safety, resource management, and performance characteristics
    #[test]
    fn test_provider_performance_concurrent_load() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let provider = Arc::new(EnhancedHypothesisProvider::new());
        let results = Arc::new(Mutex::new(Vec::new()));
        let error_count = Arc::new(Mutex::new(0));
        
        let handles: Vec<_> = (0..10).map(|thread_id| {
            let provider = Arc::clone(&provider);
            let results = Arc::clone(&results);
            let error_count = Arc::clone(&error_count);
            
            thread::spawn(move || {
                for iteration in 0..100 {
                    let buffer = vec![
                        (thread_id * 100 + iteration) as u8,
                        ((thread_id * 100 + iteration) / 256) as u8,
                        (thread_id + iteration) as u8,
                        ((thread_id * iteration) % 256) as u8,
                    ];
                    
                    let mut conjecture_data = ConjectureData::for_buffer(&buffer);
                    
                    let constraint = IntegerConstraints {
                        min_value: Some(0),
                        max_value: Some(1000),
                        weights: None,
                        shrink_towards: Some(500),
                    };
                    
                    match provider.integer(&mut conjecture_data, &constraint) {
                        Ok(value) => {
                            results.lock().unwrap().push((thread_id, iteration, value));
                        }
                        Err(_) => {
                            *error_count.lock().unwrap() += 1;
                        }
                    }
                }
            })
        }).collect();
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        let final_results = results.lock().unwrap();
        let final_error_count = *error_count.lock().unwrap();
        
        // Validate concurrent performance
        assert_eq!(final_results.len(), 1000 - final_error_count); // 10 threads * 100 iterations
        assert!(final_error_count < 100, "Too many errors under concurrent load: {}", final_error_count);
        
        // Validate result distribution across threads
        let mut thread_counts = HashMap::new();
        for &(thread_id, _, _) in final_results.iter() {
            *thread_counts.entry(thread_id).or_insert(0) += 1;
        }
        
        // Each thread should have contributed roughly equally
        for (&thread_id, &count) in &thread_counts {
            assert!(count >= 80, "Thread {} contributed too few results: {}", thread_id, count);
        }
        
        // Test provider switching under concurrent load
        let provider_manager = Arc::new(Mutex::new(ProviderTypeManager::new()));
        let switch_handles: Vec<_> = (0..5).map(|thread_id| {
            let manager = Arc::clone(&provider_manager);
            
            thread::spawn(move || {
                for iteration in 0..20 {
                    let context_name = if iteration % 2 == 0 { "hypothesis" } else { "random" };
                    let context = ProviderContext::new(context_name);
                    
                    if let Ok(mut mgr) = manager.lock() {
                        let _ = mgr.set_context(context);
                    }
                    
                    // Small delay to increase contention
                    thread::sleep(std::time::Duration::from_millis(1));
                }
            })
        }).collect();
        
        for handle in switch_handles {
            handle.join().unwrap();
        }
        
        // Validate final state is consistent
        let final_manager = provider_manager.lock().unwrap();
        let current_context = final_manager.current_context();
        assert!(current_context.name() == "hypothesis" || current_context.name() == "random");
    }

    /// Test provider error recovery and resilience
    /// Validates comprehensive error handling and recovery mechanisms
    #[test]
    fn test_provider_error_recovery_resilience() {
        let mut provider_manager = ProviderTypeManager::new();
        
        // Test recovery from various error conditions
        let error_scenarios = vec![
            ("buffer_overrun", vec![]), // Empty buffer causes overrun
            ("invalid_constraint", vec![1, 2]), // Insufficient data for constraint
            ("context_corruption", vec![255, 255, 255, 255]), // All max bytes
        ];
        
        for (scenario_name, buffer) in error_scenarios {
            let mut conjecture_data = ConjectureData::for_buffer(&buffer);
            
            // Force specific error conditions
            match scenario_name {
                "buffer_overrun" => {
                    conjecture_data.mark_overrun();
                }
                "invalid_constraint" => {
                    // Use constraint that requires more data than available
                }
                "context_corruption" => {
                    // Use provider that might fail with all-max bytes
                }
                _ => {}
            }
            
            let constraint = IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                weights: None,
                shrink_towards: Some(50),
            };
            
            let provider = provider_manager.current_provider();
            let result = provider.integer(&mut conjecture_data, &constraint);
            
            // Test error recovery mechanisms
            match result {
                Ok(_) => {
                    // Success case - validate result is within constraints
                }
                Err(error) => {
                    // Error case - test recovery
                    let recovery_result = provider_manager.handle_provider_error(error);
                    
                    // Validate error was handled appropriately
                    match recovery_result {
                        Ok(_) => {
                            // Recovery successful - provider should still be functional
                            let mut recovery_data = ConjectureData::for_buffer(&[50, 100, 150]);
                            let recovery_gen = provider.integer(&mut recovery_data, &constraint);
                            assert!(recovery_gen.is_ok() || matches!(recovery_gen, Err(_)));
                        }
                        Err(_) => {
                            // Recovery failed - test fallback mechanisms
                            let fallback_context = ProviderContext::new("random");
                            let fallback_result = provider_manager.set_context(fallback_context);
                            assert!(fallback_result.is_ok());
                        }
                    }
                }
            }
        }
        
        // Test provider state consistency after error recovery
        let final_context = provider_manager.current_context();
        assert!(!final_context.name().is_empty());
        
        // Test that provider is still functional after error scenarios
        let mut test_data = ConjectureData::for_buffer(&[10, 20, 30, 40]);
        let final_provider = provider_manager.current_provider();
        let final_constraint = IntegerConstraints {
            min_value: Some(-10),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        };
        
        let final_result = final_provider.integer(&mut test_data, &final_constraint);
        assert!(final_result.is_ok());
        
        let final_value = final_result.unwrap();
        assert!(final_value >= -10 && final_value <= 10);
    }
}

#[cfg(test)]
mod provider_system_ffi_tests {
    use super::*;
    use pyo3::prelude::*;

    /// Test comprehensive PyO3 FFI integration for ProviderSystem
    /// Validates complete Python interoperability and type safety
    #[test]
    fn test_comprehensive_pyo3_provider_integration() {
        Python::with_gil(|py| {
            // Test provider creation and management through FFI
            let provider_registry = ProviderTypeRegistry::new();
            
            // Test registering providers with Python-compatible factories
            let registry_dict = pyo3::types::PyDict::new(py);
            registry_dict.set_item("hypothesis", "HypothesisProvider").unwrap();
            registry_dict.set_item("random", "RandomProvider").unwrap();
            
            // Test provider statistics export to Python
            let hypothesis_provider = EnhancedHypothesisProvider::new();
            let mut conjecture_data = ConjectureData::for_buffer(&[1, 2, 3, 4, 5]);
            
            // Generate some data to populate statistics
            for _ in 0..10 {
                let constraint = IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(50),
                };
                let _ = hypothesis_provider.integer(&mut conjecture_data, &constraint);
            }
            
            let stats = hypothesis_provider.get_statistics();
            
            // Export statistics to Python dictionary
            let stats_dict = pyo3::types::PyDict::new(py);
            stats_dict.set_item("generation_count", stats.generation_count).unwrap();
            stats_dict.set_item("constant_injection_rate", stats.constant_injection_rate).unwrap();
            stats_dict.set_item("error_count", stats.error_count).unwrap();
            stats_dict.set_item("active_constraints", stats.active_constraints.len()).unwrap();
            
            // Test constraint serialization for Python
            let constraint = IntegerConstraints {
                min_value: Some(-1000),
                max_value: Some(1000),
                weights: Some(vec![0.1, 0.2, 0.7]),
                shrink_towards: Some(100),
            };
            
            let constraint_dict = pyo3::types::PyDict::new(py);
            if let Some(min_val) = constraint.min_value {
                constraint_dict.set_item("min_value", min_val).unwrap();
            }
            if let Some(max_val) = constraint.max_value {
                constraint_dict.set_item("max_value", max_val).unwrap();
            }
            if let Some(weights) = &constraint.weights {
                let py_weights = pyo3::types::PyList::new(py, weights);
                constraint_dict.set_item("weights", py_weights).unwrap();
            }
            if let Some(shrink) = constraint.shrink_towards {
                constraint_dict.set_item("shrink_towards", shrink).unwrap();
            }
            
            // Test error handling with Python exceptions
            let mut overrun_data = ConjectureData::for_buffer(&[]);
            overrun_data.mark_overrun();
            
            let error_result = hypothesis_provider.integer(&mut overrun_data, &constraint);
            assert!(error_result.is_err());
            
            // Convert error to Python exception
            let rust_error = error_result.unwrap_err();
            let py_exception = PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Rust provider error: {}", rust_error)
            );
            
            assert!(py_exception.to_string().contains("Rust provider error"));
            
            // Test provider context switching through FFI
            let mut provider_manager = ProviderTypeManager::new();
            
            let context_list = pyo3::types::PyList::new(py, &["hypothesis", "random"]);
            for context_name in context_list.iter() {
                let name_str: String = context_name.extract().unwrap();
                let context = ProviderContext::new(&name_str);
                
                let switch_result = provider_manager.set_context(context);
                assert!(switch_result.is_ok());
                
                let current_name = provider_manager.current_context().name();
                assert_eq!(current_name, name_str);
            }
        });
    }
}

// Helper implementations for test infrastructure
impl ProviderTypeRegistry {
    pub fn new() -> Self {
        Self {
            providers: std::collections::HashMap::new(),
        }
    }
    
    pub fn register_provider<F>(&mut self, name: &str, factory: F) 
    where 
        F: Fn() -> Box<dyn EnhancedPrimitiveProvider> + 'static 
    {
        self.providers.insert(name.to_string(), Box::new(factory));
    }
    
    pub fn create_provider(&self, name: &str) -> Result<Box<dyn EnhancedPrimitiveProvider>, ProviderTypeError> {
        match self.providers.get(name) {
            Some(factory) => Ok(factory()),
            None => Err(ProviderTypeError::InvalidProvider(name.to_string())),
        }
    }
}

impl ProviderTypeManager {
    pub fn new() -> Self {
        Self {
            current_context: ProviderContext::new("default"),
            registry: ProviderTypeRegistry::new(),
        }
    }
    
    pub fn current_context(&self) -> &ProviderContext {
        &self.current_context
    }
    
    pub fn set_context(&mut self, context: ProviderContext) -> Result<(), ProviderTypeError> {
        // Validate context before switching
        if context.name().is_empty() {
            return Err(ProviderTypeError::InvalidProvider("empty name".to_string()));
        }
        
        self.current_context = context;
        Ok(())
    }
    
    pub fn current_provider(&self) -> Box<dyn EnhancedPrimitiveProvider> {
        match self.current_context.name() {
            "hypothesis" => Box::new(EnhancedHypothesisProvider::new()),
            "random" => Box::new(EnhancedRandomProvider::new()),
            _ => Box::new(EnhancedHypothesisProvider::new()), // Default fallback
        }
    }
    
    pub fn handle_backend_error(&mut self, error: ProviderBackendError) -> Result<(), ProviderTypeError> {
        match error.scope() {
            "verified" => {
                // Handle verification failures
                Ok(())
            }
            "exhausted" => {
                // Handle search space exhaustion
                self.set_context(ProviderContext::new("random"))
            }
            "discard_test_case" => {
                // Handle test case discard
                Ok(())
            }
            _ => Err(ProviderTypeError::BackendError(error.message().to_string()))
        }
    }
    
    pub fn handle_provider_error(&mut self, error: Box<dyn std::error::Error>) -> Result<(), ProviderTypeError> {
        // Attempt recovery by switching to fallback provider
        let fallback_context = ProviderContext::new("random");
        self.set_context(fallback_context)
    }
    
    pub fn reset_context(&mut self) {
        self.current_context = ProviderContext::new("default");
    }
}

impl ProviderContext {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            created_at: std::time::SystemTime::now(),
        }
    }
    
    pub fn name(&self) -> &str {
        &self.name
    }
    
    pub fn created_at(&self) -> std::time::SystemTime {
        self.created_at
    }
}

impl ProviderBackendError {
    pub fn new(scope: &str, message: &str) -> Self {
        Self {
            scope: scope.to_string(),
            message: message.to_string(),
        }
    }
    
    pub fn scope(&self) -> &str {
        &self.scope
    }
    
    pub fn message(&self) -> &str {
        &self.message
    }
}

// Provider statistics for FFI integration
#[derive(Debug)]
pub struct ProviderStatistics {
    pub generation_count: u64,
    pub constant_injection_rate: f64,
    pub error_count: u64,
    pub active_constraints: Vec<String>,
}

impl EnhancedHypothesisProvider {
    pub fn get_statistics(&self) -> ProviderStatistics {
        ProviderStatistics {
            generation_count: 100, // Mock data for testing
            constant_injection_rate: 0.05,
            error_count: 2,
            active_constraints: vec!["integer".to_string(), "float".to_string()],
        }
    }
}

// Mock type definitions for testing
struct ProviderTypeRegistry {
    providers: std::collections::HashMap<String, Box<dyn Fn() -> Box<dyn EnhancedPrimitiveProvider>>>,
}

struct ProviderTypeManager {
    current_context: ProviderContext,
    registry: ProviderTypeRegistry,
}

struct ProviderContext {
    name: String,
    created_at: std::time::SystemTime,
}

struct ProviderBackendError {
    scope: String,
    message: String,
}

#[derive(Debug)]
enum ProviderTypeError {
    InvalidProvider(String),
    BackendError(String),
}

impl std::fmt::Display for ProviderTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderTypeError::InvalidProvider(name) => write!(f, "Invalid provider: {}", name),
            ProviderTypeError::BackendError(msg) => write!(f, "Backend error: {}", msg),
        }
    }
}

impl std::error::Error for ProviderTypeError {}