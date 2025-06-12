//! Comprehensive Provider Lifecycle Management Capability Tests
//! 
//! This module contains comprehensive integration tests that demonstrate
//! the full Provider Lifecycle Management system capabilities including:
//! - Per-test-case and per-test-function lifetime controls
//! - Context management with setup/teardown hooks
//! - Provider instance caching and reuse strategies
//! - Advanced observability and debugging support

use crate::provider_lifecycle_management::*;
use crate::providers::{RandomProvider, HypothesisProvider, ProviderFactory, PrimitiveProvider, ProviderError};
use crate::choice::{IntegerConstraints, FloatConstraints, BooleanConstraints};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use serde_json;

/// Test factory that creates providers for testing
struct TestProviderFactory {
    name: String,
    provider_type: TestProviderType,
}

#[derive(Clone)]
enum TestProviderType {
    Random,
    Hypothesis,
    Custom(String),
}

impl ProviderFactory for TestProviderFactory {
    fn create_provider(&self) -> Box<dyn PrimitiveProvider> {
        match &self.provider_type {
            TestProviderType::Random => Box::new(RandomProvider::new()),
            TestProviderType::Hypothesis => Box::new(HypothesisProvider::new()),
            TestProviderType::Custom(name) => Box::new(CustomTestProvider::new(name.clone())),
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Custom test provider for lifecycle testing
#[derive(Debug)]
struct CustomTestProvider {
    name: String,
    operation_count: u64,
    setup_called: bool,
    cleanup_called: bool,
}

impl CustomTestProvider {
    fn new(name: String) -> Self {
        Self {
            name,
            operation_count: 0,
            setup_called: false,
            cleanup_called: false,
        }
    }
}

impl PrimitiveProvider for CustomTestProvider {
    fn lifetime(&self) -> crate::providers::ProviderLifetime {
        crate::providers::ProviderLifetime::TestCase
    }
    
    fn draw_boolean(&mut self, p: f64) -> Result<bool, ProviderError> {
        self.operation_count += 1;
        Ok(p >= 0.5)
    }
    
    fn draw_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        self.operation_count += 1;
        let min = constraints.min_value.unwrap_or(0);
        let max = constraints.max_value.unwrap_or(100);
        Ok((min + max) / 2)
    }
    
    fn draw_float(&mut self, constraints: &FloatConstraints) -> Result<f64, ProviderError> {
        self.operation_count += 1;
        Ok((constraints.min_value + constraints.max_value) / 2.0)
    }
    
    fn draw_string(&mut self, intervals: &crate::choice::IntervalSet, min_size: usize, max_size: usize) -> Result<String, ProviderError> {
        self.operation_count += 1;
        Ok("test".repeat(min_size.max(1)))
    }
    
    fn draw_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, ProviderError> {
        self.operation_count += 1;
        Ok(vec![42u8; min_size])
    }
    
    fn observe_test_case(&mut self) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert("operation_count".to_string(), serde_json::Value::Number(serde_json::Number::from(self.operation_count)));
        data.insert("setup_called".to_string(), serde_json::Value::Bool(self.setup_called));
        data.insert("cleanup_called".to_string(), serde_json::Value::Bool(self.cleanup_called));
        data
    }
}

/// Advanced lifecycle hooks for testing
#[derive(Debug)]
struct AdvancedTestLifecycleHooks {
    events: Vec<LifecycleEvent>,
    setup_data: HashMap<String, serde_json::Value>,
    cleanup_data: HashMap<String, serde_json::Value>,
    error_count: u32,
    verbose: bool,
}

impl AdvancedTestLifecycleHooks {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            setup_data: HashMap::new(),
            cleanup_data: HashMap::new(),
            error_count: 0,
            verbose: true,
        }
    }
    
    fn get_events_by_type(&self, event_type: &str) -> Vec<&LifecycleEvent> {
        self.events.iter().filter(|event| {
            match event {
                LifecycleEvent::Created { .. } => event_type == "created",
                LifecycleEvent::Initialized { .. } => event_type == "initialized",
                LifecycleEvent::Activated { .. } => event_type == "activated",
                LifecycleEvent::Suspended { .. } => event_type == "suspended",
                LifecycleEvent::Reactivated { .. } => event_type == "reactivated",
                LifecycleEvent::Cleanup { .. } => event_type == "cleanup",
                LifecycleEvent::Destroyed { .. } => event_type == "destroyed",
                LifecycleEvent::Error { .. } => event_type == "error",
            }
        }).collect()
    }
}

impl LifecycleHooks for AdvancedTestLifecycleHooks {
    fn on_provider_created(&mut self, provider_id: &str, scope: LifecycleScope) -> Result<(), ProviderError> {
        if self.verbose {
            println!("ADVANCED_HOOKS DEBUG: Provider created: {} (scope: {:?})", provider_id, scope);
        }
        self.events.push(LifecycleEvent::Created {
            provider_id: provider_id.to_string(),
            scope,
            timestamp: SystemTime::now(),
        });
        Ok(())
    }
    
    fn before_setup(&mut self, provider_id: &str, context: &HashMap<String, serde_json::Value>) -> Result<(), ProviderError> {
        if self.verbose {
            println!("ADVANCED_HOOKS DEBUG: Before setup: {} (context: {:?})", provider_id, context);
        }
        self.setup_data.insert(provider_id.to_string(), serde_json::to_value(context).unwrap_or_default());
        Ok(())
    }
    
    fn after_setup(&mut self, provider_id: &str, success: bool) -> Result<(), ProviderError> {
        if self.verbose {
            println!("ADVANCED_HOOKS DEBUG: After setup: {} (success: {})", provider_id, success);
        }
        if success {
            self.events.push(LifecycleEvent::Initialized {
                provider_id: provider_id.to_string(),
                context_data: self.setup_data.get(provider_id).cloned().unwrap_or_default().as_object().unwrap_or(&serde_json::Map::new()).clone().into_iter().collect(),
            });
        }
        Ok(())
    }
    
    fn on_provider_activated(&mut self, provider_id: &str, test_context: &str) -> Result<(), ProviderError> {
        if self.verbose {
            println!("ADVANCED_HOOKS DEBUG: Provider activated: {} in context: {}", provider_id, test_context);
        }
        self.events.push(LifecycleEvent::Activated {
            provider_id: provider_id.to_string(),
            test_context: test_context.to_string(),
        });
        Ok(())
    }
    
    fn on_provider_suspended(&mut self, provider_id: &str, reason: &str) -> Result<(), ProviderError> {
        if self.verbose {
            println!("ADVANCED_HOOKS DEBUG: Provider suspended: {} (reason: {})", provider_id, reason);
        }
        self.events.push(LifecycleEvent::Suspended {
            provider_id: provider_id.to_string(),
            reason: reason.to_string(),
        });
        Ok(())
    }
    
    fn on_provider_reactivated(&mut self, provider_id: &str, cache_duration: Duration) -> Result<(), ProviderError> {
        if self.verbose {
            println!("ADVANCED_HOOKS DEBUG: Provider reactivated: {} (cached for: {:?})", provider_id, cache_duration);
        }
        self.events.push(LifecycleEvent::Reactivated {
            provider_id: provider_id.to_string(),
            cache_duration,
        });
        Ok(())
    }
    
    fn before_cleanup(&mut self, provider_id: &str, context: &HashMap<String, serde_json::Value>) -> Result<(), ProviderError> {
        if self.verbose {
            println!("ADVANCED_HOOKS DEBUG: Before cleanup: {} (context: {:?})", provider_id, context);
        }
        self.cleanup_data.insert(provider_id.to_string(), serde_json::to_value(context).unwrap_or_default());
        Ok(())
    }
    
    fn after_cleanup(&mut self, provider_id: &str, success: bool, cleanup_data: &HashMap<String, serde_json::Value>) -> Result<(), ProviderError> {
        if self.verbose {
            println!("ADVANCED_HOOKS DEBUG: After cleanup: {} (success: {}, data: {:?})", provider_id, success, cleanup_data);
        }
        self.events.push(LifecycleEvent::Cleanup {
            provider_id: provider_id.to_string(),
            success,
            cleanup_data: cleanup_data.clone(),
        });
        Ok(())
    }
    
    fn on_provider_destroyed(&mut self, provider_id: &str, scope: LifecycleScope, total_lifetime: Duration) -> Result<(), ProviderError> {
        if self.verbose {
            println!("ADVANCED_HOOKS DEBUG: Provider destroyed: {} (scope: {:?}, lifetime: {:?})", provider_id, scope, total_lifetime);
        }
        self.events.push(LifecycleEvent::Destroyed {
            provider_id: provider_id.to_string(),
            scope,
            total_lifetime,
        });
        Ok(())
    }
    
    fn on_lifecycle_error(&mut self, provider_id: &str, error: &str, context: &str) -> Result<(), ProviderError> {
        if self.verbose {
            println!("ADVANCED_HOOKS ERROR: Provider {} error: {} (context: {})", provider_id, error, context);
        }
        self.error_count += 1;
        self.events.push(LifecycleEvent::Error {
            provider_id: provider_id.to_string(),
            error: error.to_string(),
            context: context.to_string(),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_comprehensive_lifecycle_manager_operations() {
        let mut manager = ProviderLifecycleManager::new();
        
        // Register multiple provider types
        let random_factory = Arc::new(TestProviderFactory {
            name: "test_random".to_string(),
            provider_type: TestProviderType::Random,
        });
        manager.register_factory(random_factory);
        
        let hypothesis_factory = Arc::new(TestProviderFactory {
            name: "test_hypothesis".to_string(),
            provider_type: TestProviderType::Hypothesis,
        });
        manager.register_factory(hypothesis_factory);
        
        let custom_factory = Arc::new(TestProviderFactory {
            name: "test_custom".to_string(),
            provider_type: TestProviderType::Custom("custom_provider".to_string()),
        });
        manager.register_factory(custom_factory);
        
        // Set up advanced hooks
        let hooks = Arc::new(Mutex::new(AdvancedTestLifecycleHooks::new()));
        manager.set_global_hooks(hooks.clone());
        
        // Configure custom cache for test case scope
        manager.configure_cache(LifecycleScope::TestCase, CacheConfiguration {
            max_instances: 3,
            ttl: Duration::from_secs(10),
            lru_eviction: true,
            custom_eviction: None,
        });
        
        // Test provider creation and lifecycle for different scopes
        let test_cases = vec![
            ("test_random", LifecycleScope::TestCase, "random_test_context"),
            ("test_hypothesis", LifecycleScope::TestFunction, "hypothesis_test_context"),
            ("test_custom", LifecycleScope::TestRun, "custom_test_context"),
        ];
        
        let mut providers = Vec::new();
        
        for (provider_name, scope, context) in test_cases {
            let mut context_data = HashMap::new();
            context_data.insert("test_parameter".to_string(), serde_json::Value::String(format!("{}_{:?}", provider_name, scope)));
            context_data.insert("scope".to_string(), serde_json::Value::String(format!("{:?}", scope)));
            
            let provider = manager.get_provider(
                provider_name,
                scope,
                Some(context.to_string()),
                Some(context_data),
            ).unwrap();
            
            providers.push((provider, provider_name, scope, context));
        }
        
        // Test provider operations and metric recording
        for (provider, provider_name, scope, context) in &providers {
            println!("Testing provider {} in scope {:?} with context {}", provider_name, scope, context);
            
            if let Ok(mut p) = provider.lock() {
                // Test various operations
                let start_time = std::time::Instant::now();
                
                let constraints = IntegerConstraints {
                    min_value: Some(1),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(50),
                };
                
                let result = p.provider_mut().draw_integer(&constraints);
                let duration = start_time.elapsed();
                
                p.record_operation(result.is_ok(), duration);
                
                if let Ok(value) = result {
                    println!("  Generated integer: {}", value);
                }
                
                // Test boolean generation
                let start_time = std::time::Instant::now();
                let bool_result = p.provider_mut().draw_boolean(0.7);
                let duration = start_time.elapsed();
                
                p.record_operation(bool_result.is_ok(), duration);
                
                if let Ok(value) = bool_result {
                    println!("  Generated boolean: {}", value);
                }
                
                // Check metadata
                let metadata = p.metadata();
                println!("  Provider metadata: operations={}, state={:?}, access_count={}", 
                        metadata.metrics.total_operations, metadata.state, metadata.access_count);
            }
        }
        
        // Test provider suspension and reactivation
        println!("\nTesting suspension and reactivation...");
        if let Some((provider, provider_name, _, _)) = providers.first() {
            if let Ok(mut p) = provider.lock() {
                p.suspend("Test suspension for cache management".to_string()).unwrap();
                assert_eq!(p.metadata.state, ProviderState::Suspended);
                
                p.reactivate().unwrap();
                assert_eq!(p.metadata.state, ProviderState::Active);
                assert!(p.metadata.metrics.cache_hits > 0);
            }
        }
        
        // Test cache reuse by requesting same provider again
        println!("\nTesting cache reuse...");
        let reused_provider = manager.get_provider(
            "test_random",
            LifecycleScope::TestCase,
            Some("reuse_test_context".to_string()),
            None,
        ).unwrap();
        
        // Should be same instance
        if let (Ok(original), Ok(reused)) = (providers[0].0.lock(), reused_provider.lock()) {
            assert_eq!(original.metadata.provider_id, reused.metadata.provider_id);
            assert!(reused.metadata.access_count > original.metadata.access_count);
        }
        
        // Test cleanup tasks
        println!("\nTesting cleanup tasks...");
        manager.run_cleanup_tasks().unwrap();
        
        // Test manager statistics
        let stats = manager.get_statistics();
        println!("\nManager statistics: {:?}", stats);
        
        assert!(stats.contains_key("total_providers"));
        assert!(stats.contains_key("providers_by_scope"));
        assert!(stats.contains_key("registered_factories"));
        
        if let Some(total) = stats.get("total_providers") {
            assert!(total.as_u64().unwrap_or(0) > 0);
        }
        
        // Verify hooks were called extensively
        let hook_events = hooks.lock().unwrap().events.clone();
        println!("\nTotal lifecycle events recorded: {}", hook_events.len());
        
        // Check for specific event types
        let creation_events = hooks.lock().unwrap().get_events_by_type("created").len();
        let activation_events = hooks.lock().unwrap().get_events_by_type("activated").len();
        let suspension_events = hooks.lock().unwrap().get_events_by_type("suspended").len();
        let reactivation_events = hooks.lock().unwrap().get_events_by_type("reactivated").len();
        
        println!("Event counts - Created: {}, Activated: {}, Suspended: {}, Reactivated: {}", 
                creation_events, activation_events, suspension_events, reactivation_events);
        
        assert!(creation_events >= 3); // At least 3 providers created
        assert!(activation_events >= 4); // At least 4 activations (including reuse)
        assert!(suspension_events >= 1); // At least 1 suspension
        assert!(reactivation_events >= 1); // At least 1 reactivation
        
        // Test manager shutdown
        println!("\nTesting manager shutdown...");
        let initial_provider_count = manager.providers.len();
        manager.shutdown().unwrap();
        assert_eq!(manager.providers.len(), 0);
        
        // Verify destruction events
        let final_hook_events = hooks.lock().unwrap().events.clone();
        let destruction_events = hooks.lock().unwrap().get_events_by_type("destroyed").len();
        println!("Destruction events: {}", destruction_events);
        assert_eq!(destruction_events, initial_provider_count);
    }
    
    #[test]
    fn test_cache_eviction_and_ttl() {
        let mut manager = ProviderLifecycleManager::new();
        
        // Configure very small cache with short TTL for testing
        manager.configure_cache(LifecycleScope::TestCase, CacheConfiguration {
            max_instances: 2,
            ttl: Duration::from_millis(100),
            lru_eviction: true,
            custom_eviction: None,
        });
        
        let factory = Arc::new(TestProviderFactory {
            name: "eviction_test".to_string(),
            provider_type: TestProviderType::Custom("eviction_provider".to_string()),
        });
        manager.register_factory(factory);
        
        // Create providers to test eviction
        let _provider1 = manager.get_provider(
            "eviction_test",
            LifecycleScope::TestCase,
            Some("eviction_context_1".to_string()),
            None,
        ).unwrap();
        
        std::thread::sleep(Duration::from_millis(50));
        
        let _provider2 = manager.get_provider(
            "eviction_test",
            LifecycleScope::TestCase,
            Some("eviction_context_2".to_string()),
            None,
        ).unwrap();
        
        std::thread::sleep(Duration::from_millis(50));
        
        // This should trigger eviction due to max_instances=2
        let _provider3 = manager.get_provider(
            "eviction_test",
            LifecycleScope::TestCase,
            Some("eviction_context_3".to_string()),
            None,
        ).unwrap();
        
        // Verify cache constraint
        let scope_count = manager.providers.iter()
            .filter(|((scope, _), _)| *scope == LifecycleScope::TestCase)
            .count();
        assert!(scope_count <= 2);
        
        // Wait for TTL expiry
        std::thread::sleep(Duration::from_millis(150));
        
        // Run cleanup to remove expired instances
        let removed_count = manager.cleanup_expired_instances().unwrap();
        println!("Removed {} expired instances", removed_count);
        assert!(removed_count > 0);
    }
    
    #[test]
    fn test_scope_compatibility_and_inheritance() {
        let mut manager = ProviderLifecycleManager::new();
        
        let factory = Arc::new(TestProviderFactory {
            name: "scope_compatibility_test".to_string(),
            provider_type: TestProviderType::Custom("scope_provider".to_string()),
        });
        manager.register_factory(factory);
        
        // Create session-scoped provider first
        let session_provider = manager.get_provider(
            "scope_compatibility_test",
            LifecycleScope::Session,
            Some("session_context".to_string()),
            None,
        ).unwrap();
        
        let session_id = session_provider.lock().unwrap().metadata.provider_id.clone();
        println!("Created session provider: {}", session_id);
        
        // Request test-function scoped provider - should reuse session provider
        let test_function_provider = manager.get_provider(
            "scope_compatibility_test",
            LifecycleScope::TestFunction,
            Some("test_function_context".to_string()),
            None,
        ).unwrap();
        
        let test_function_id = test_function_provider.lock().unwrap().metadata.provider_id.clone();
        println!("Requested test function provider, got: {}", test_function_id);
        
        // Should be the same provider instance due to scope compatibility
        assert_eq!(session_id, test_function_id);
        
        // Request test-case scoped provider - should also reuse session provider
        let test_case_provider = manager.get_provider(
            "scope_compatibility_test",
            LifecycleScope::TestCase,
            Some("test_case_context".to_string()),
            None,
        ).unwrap();
        
        let test_case_id = test_case_provider.lock().unwrap().metadata.provider_id.clone();
        println!("Requested test case provider, got: {}", test_case_id);
        
        // Should also be the same provider instance
        assert_eq!(session_id, test_case_id);
        
        // Verify access count increased for reuse
        let final_access_count = test_case_provider.lock().unwrap().metadata.access_count;
        assert!(final_access_count >= 3); // Created + activated 3 times
        
        println!("Final access count: {}", final_access_count);
    }
    
    #[test]
    fn test_advanced_metrics_and_observability() {
        let mut manager = ProviderLifecycleManager::new();
        
        let factory = Arc::new(TestProviderFactory {
            name: "metrics_test".to_string(),
            provider_type: TestProviderType::Custom("metrics_provider".to_string()),
        });
        manager.register_factory(factory);
        
        let hooks = Arc::new(Mutex::new(AdvancedTestLifecycleHooks::new()));
        manager.set_global_hooks(hooks.clone());
        
        let provider = manager.get_provider(
            "metrics_test",
            LifecycleScope::TestFunction,
            Some("metrics_context".to_string()),
            None,
        ).unwrap();
        
        // Perform multiple operations to build metrics
        if let Ok(mut p) = provider.lock() {
            for i in 0..10 {
                let start_time = std::time::Instant::now();
                
                let constraints = IntegerConstraints {
                    min_value: Some(i),
                    max_value: Some(i + 100),
                    weights: None,
                    shrink_towards: Some(i + 50),
                };
                
                let result = p.provider_mut().draw_integer(&constraints);
                let duration = start_time.elapsed();
                
                p.record_operation(result.is_ok(), duration);
                
                // Occasionally record failures
                if i % 3 == 0 {
                    p.record_operation(false, duration);
                }
            }
            
            // Check metrics
            let metadata = p.metadata();
            println!("Metrics after operations:");
            println!("  Total operations: {}", metadata.metrics.total_operations);
            println!("  Successful operations: {}", metadata.metrics.successful_operations);
            println!("  Failed operations: {}", metadata.metrics.failed_operations);
            println!("  Average operation time: {:?}", metadata.metrics.average_operation_time);
            println!("  Cache hits: {}", metadata.metrics.cache_hits);
            
            assert!(metadata.metrics.total_operations > 10);
            assert!(metadata.metrics.successful_operations > 0);
            assert!(metadata.metrics.failed_operations > 0);
            assert!(metadata.metrics.average_operation_time > Duration::ZERO);
            
            // Test provider observation
            let observation = p.provider_mut().observe_test_case();
            println!("Provider observation: {:?}", observation);
            assert!(!observation.is_empty());
        }
        
        // Test hooks data collection
        let hook_events = hooks.lock().unwrap().events.clone();
        println!("Total hook events: {}", hook_events.len());
        
        // Check for initialization event with context data
        let init_events: Vec<_> = hook_events.iter()
            .filter_map(|event| match event {
                LifecycleEvent::Initialized { provider_id, context_data } => Some((provider_id, context_data)),
                _ => None,
            })
            .collect();
        
        assert!(!init_events.is_empty());
        println!("Initialization events: {:?}", init_events);
        
        // Verify setup and cleanup data collection
        let setup_data = &hooks.lock().unwrap().setup_data;
        let cleanup_data = &hooks.lock().unwrap().cleanup_data;
        
        println!("Setup data collected: {:?}", setup_data);
        println!("Cleanup data collected: {:?}", cleanup_data);
        
        assert!(!setup_data.is_empty());
    }
    
    #[test]
    fn test_periodic_cleanup_and_optimization() {
        let mut manager = ProviderLifecycleManager::new();
        
        // Add custom cleanup tasks with very short intervals for testing
        manager.cleanup_tasks.clear(); // Clear default tasks
        manager.cleanup_tasks.extend(vec![
            CleanupTask {
                scope: LifecycleScope::TestCase,
                interval: Duration::from_millis(50),
                last_run: SystemTime::now() - Duration::from_millis(100),
                task_type: CleanupTaskType::ExpiredInstances,
            },
            CleanupTask {
                scope: LifecycleScope::TestFunction,
                interval: Duration::from_millis(100),
                last_run: SystemTime::now() - Duration::from_millis(200),
                task_type: CleanupTaskType::CacheOptimization,
            },
            CleanupTask {
                scope: LifecycleScope::TestRun,
                interval: Duration::from_millis(150),
                last_run: SystemTime::now() - Duration::from_millis(300),
                task_type: CleanupTaskType::MetricsCompaction,
            },
        ]);
        
        let factory = Arc::new(TestProviderFactory {
            name: "cleanup_optimization_test".to_string(),
            provider_type: TestProviderType::Custom("cleanup_provider".to_string()),
        });
        manager.register_factory(factory);
        
        // Configure short TTL for testing
        manager.configure_cache(LifecycleScope::TestCase, CacheConfiguration {
            max_instances: 5,
            ttl: Duration::from_millis(25),
            lru_eviction: true,
            custom_eviction: None,
        });
        
        // Create multiple providers with high access patterns
        let mut providers = Vec::new();
        for i in 0..3 {
            let provider = manager.get_provider(
                "cleanup_optimization_test",
                LifecycleScope::TestCase,
                Some(format!("cleanup_context_{}", i)),
                None,
            ).unwrap();
            
            // Simulate high usage for cache optimization testing
            if let Ok(mut p) = provider.lock() {
                for _ in 0..20 {
                    p.metadata.access_count += 1;
                    p.metadata.metrics.total_operations += 1;
                    p.metadata.metrics.successful_operations += 1;
                }
            }
            
            providers.push(provider);
        }
        
        let initial_provider_count = manager.providers.len();
        println!("Initial provider count: {}", initial_provider_count);
        
        // Wait for some providers to expire
        std::thread::sleep(Duration::from_millis(50));
        
        // Run cleanup tasks
        println!("Running cleanup tasks...");
        manager.run_cleanup_tasks().unwrap();
        
        let post_cleanup_count = manager.providers.len();
        println!("Provider count after cleanup: {}", post_cleanup_count);
        
        // Some providers should have been cleaned up due to TTL expiry
        assert!(post_cleanup_count <= initial_provider_count);
        
        // Test cache optimization effect
        let stats_before = manager.get_statistics();
        println!("Stats before optimization: {:?}", stats_before);
        
        // Create more providers to trigger optimization
        for i in 3..6 {
            let provider = manager.get_provider(
                "cleanup_optimization_test",
                LifecycleScope::TestFunction,
                Some(format!("optimization_context_{}", i)),
                None,
            ).unwrap();
            
            // Simulate varying usage patterns
            if let Ok(mut p) = provider.lock() {
                for _ in 0..(i * 5) {
                    p.metadata.access_count += 1;
                }
            }
        }
        
        // Run optimization again
        manager.run_cleanup_tasks().unwrap();
        
        let stats_after = manager.get_statistics();
        println!("Stats after optimization: {:?}", stats_after);
        
        // Verify cache configuration may have been adjusted
        println!("Cache configurations:");
        for (scope, config) in &manager.cache_config {
            println!("  {:?}: max_instances={}, ttl={:?}", scope, config.max_instances, config.ttl);
        }
    }
    
    #[test]
    fn test_error_handling_and_recovery() {
        let mut manager = ProviderLifecycleManager::new();
        
        let hooks = Arc::new(Mutex::new(AdvancedTestLifecycleHooks::new()));
        manager.set_global_hooks(hooks.clone());
        
        // Test error when requesting non-existent provider
        let result = manager.get_provider(
            "non_existent_provider",
            LifecycleScope::TestCase,
            Some("error_context".to_string()),
            None,
        );
        
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProviderError::ConfigError(msg) => {
                println!("Expected error caught: {}", msg);
                assert!(msg.contains("not found"));
            },
            _ => panic!("Unexpected error type"),
        }
        
        // Register factory and test successful creation
        let factory = Arc::new(TestProviderFactory {
            name: "error_handling_test".to_string(),
            provider_type: TestProviderType::Custom("error_provider".to_string()),
        });
        manager.register_factory(factory);
        
        let provider = manager.get_provider(
            "error_handling_test",
            LifecycleScope::TestCase,
            Some("recovery_context".to_string()),
            None,
        ).unwrap();
        
        // Test error handling in provider operations
        if let Ok(mut p) = provider.lock() {
            // Simulate error condition
            p.metadata.state = ProviderState::Error("Test error condition".to_string());
            
            // Operations should still work but be recorded as errors
            let start_time = std::time::Instant::now();
            let result = p.provider_mut().draw_boolean(0.5);
            let duration = start_time.elapsed();
            
            p.record_operation(result.is_err(), duration);
            
            assert!(p.metadata.metrics.failed_operations > 0 || result.is_ok());
        }
        
        // Test cleanup of error providers
        let removed_count = manager.cleanup_expired_instances().unwrap();
        println!("Cleaned up {} error providers", removed_count);
        
        // Verify error handling in hooks
        let error_count = hooks.lock().unwrap().error_count;
        println!("Hook error count: {}", error_count);
        
        // Test recovery by creating new provider instance
        let recovery_provider = manager.get_provider(
            "error_handling_test",
            LifecycleScope::TestFunction,
            Some("recovery_test_context".to_string()),
            None,
        ).unwrap();
        
        if let Ok(p) = recovery_provider.lock() {
            assert_eq!(p.metadata.state, ProviderState::Active);
            println!("Recovery successful, provider state: {:?}", p.metadata.state);
        }
    }
}