//! Provider Lifecycle Management System Demonstration
//! 
//! This demo showcases the complete Provider Lifecycle Management capability
//! including per-test-case and per-test-function lifetime controls, context
//! management with setup/teardown hooks, and provider instance caching.

use conjecture::{
    ProviderLifecycleManager, ManagedProvider, LifecycleScope, LifecycleEvent, LifecycleHooks,
    DefaultLifecycleHooks, ProviderInstanceMetadata, ProviderState, ProviderMetrics,
    CacheConfiguration, CleanupTask, CleanupTaskType
};
use conjecture::{RandomProvider, HypothesisProvider, PrimitiveProvider};
use conjecture::providers::{ProviderFactory, ProviderError};
use conjecture::choice::{IntegerConstraints, FloatConstraints};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use serde_json;

/// Demo provider factory
struct DemoProviderFactory {
    name: String,
}

impl ProviderFactory for DemoProviderFactory {
    fn create_provider(&self) -> Box<dyn PrimitiveProvider> {
        match self.name.as_str() {
            "demo_random" => Box::new(RandomProvider::new()),
            "demo_hypothesis" => Box::new(HypothesisProvider::new()),
            _ => Box::new(RandomProvider::new()),
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Demo lifecycle hooks that track events
#[derive(Debug)]
struct DemoLifecycleHooks {
    events: Vec<LifecycleEvent>,
}

impl DemoLifecycleHooks {
    fn new() -> Self {
        Self {
            events: Vec::new(),
        }
    }
}

impl LifecycleHooks for DemoLifecycleHooks {
    fn on_provider_created(&mut self, provider_id: &str, scope: LifecycleScope) -> Result<(), ProviderError> {
        println!("🎯 DEMO: Provider '{}' created with scope: {:?}", provider_id, scope);
        self.events.push(LifecycleEvent::Created {
            provider_id: provider_id.to_string(),
            scope,
            timestamp: std::time::SystemTime::now(),
        });
        Ok(())
    }
    
    fn on_provider_activated(&mut self, provider_id: &str, test_context: &str) -> Result<(), ProviderError> {
        println!("🚀 DEMO: Provider '{}' activated for context: {}", provider_id, test_context);
        self.events.push(LifecycleEvent::Activated {
            provider_id: provider_id.to_string(),
            test_context: test_context.to_string(),
        });
        Ok(())
    }
    
    fn on_provider_suspended(&mut self, provider_id: &str, reason: &str) -> Result<(), ProviderError> {
        println!("⏸️  DEMO: Provider '{}' suspended: {}", provider_id, reason);
        self.events.push(LifecycleEvent::Suspended {
            provider_id: provider_id.to_string(),
            reason: reason.to_string(),
        });
        Ok(())
    }
    
    fn on_provider_reactivated(&mut self, provider_id: &str, cache_duration: Duration) -> Result<(), ProviderError> {
        println!("🔄 DEMO: Provider '{}' reactivated from cache (cached for: {:?})", provider_id, cache_duration);
        self.events.push(LifecycleEvent::Reactivated {
            provider_id: provider_id.to_string(),
            cache_duration,
        });
        Ok(())
    }
    
    fn on_provider_destroyed(&mut self, provider_id: &str, scope: LifecycleScope, total_lifetime: Duration) -> Result<(), ProviderError> {
        println!("💥 DEMO: Provider '{}' destroyed (scope: {:?}, lived for: {:?})", provider_id, scope, total_lifetime);
        self.events.push(LifecycleEvent::Destroyed {
            provider_id: provider_id.to_string(),
            scope,
            total_lifetime,
        });
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎪 Provider Lifecycle Management System Demo");
    println!("=============================================\n");
    
    // Initialize the lifecycle manager
    let mut manager = ProviderLifecycleManager::new();
    
    // Set up demo hooks
    let hooks = Arc::new(Mutex::new(DemoLifecycleHooks::new()));
    manager.set_global_hooks(hooks.clone());
    
    // Register demo provider factories
    manager.register_factory(Arc::new(DemoProviderFactory {
        name: "demo_random".to_string(),
    }));
    manager.register_factory(Arc::new(DemoProviderFactory {
        name: "demo_hypothesis".to_string(),
    }));
    
    // Configure caching for demonstration
    manager.configure_cache(LifecycleScope::TestCase, CacheConfiguration {
        max_instances: 2,
        ttl: Duration::from_secs(5),
        lru_eviction: true,
        custom_eviction: None,
    });
    
    println!("📋 Demo Step 1: Creating providers with different scopes");
    println!("-------------------------------------------------------");
    
    // Create providers with different lifecycle scopes
    let test_case_provider = manager.get_provider(
        "demo_random",
        LifecycleScope::TestCase,
        Some("test_case_demo".to_string()),
        Some({
            let mut data = HashMap::new();
            data.insert("demo_param".to_string(), serde_json::Value::String("test_value".to_string()));
            data
        }),
    )?;
    
    let test_function_provider = manager.get_provider(
        "demo_hypothesis",
        LifecycleScope::TestFunction,
        Some("test_function_demo".to_string()),
        None,
    )?;
    
    println!("\n📊 Demo Step 2: Using providers for operations");
    println!("----------------------------------------------");
    
    // Use the test case provider
    {
        let mut provider = test_case_provider.lock().unwrap();
        
        let constraints = IntegerConstraints {
            min_value: Some(1),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(50),
        };
        
        let start_time = std::time::Instant::now();
        match provider.provider_mut().draw_integer(&constraints) {
            Ok(value) => {
                provider.record_operation(true, start_time.elapsed());
                println!("✅ Test case provider generated integer: {}", value);
            },
            Err(e) => {
                provider.record_operation(false, start_time.elapsed());
                println!("❌ Test case provider error: {}", e);
            }
        }
        
        // Show metadata
        let metadata = provider.metadata();
        println!("📈 Test case provider metadata: operations={}, state={:?}", 
                metadata.metrics.total_operations, metadata.state);
    }
    
    // Use the test function provider
    {
        let mut provider = test_function_provider.lock().unwrap();
        
        let start_time = std::time::Instant::now();
        match provider.provider_mut().draw_boolean(0.7) {
            Ok(value) => {
                provider.record_operation(true, start_time.elapsed());
                println!("✅ Test function provider generated boolean: {}", value);
            },
            Err(e) => {
                provider.record_operation(false, start_time.elapsed());
                println!("❌ Test function provider error: {}", e);
            }
        }
    }
    
    println!("\n🔄 Demo Step 3: Testing provider suspension and reactivation");
    println!("------------------------------------------------------------");
    
    // Suspend provider
    {
        let mut provider = test_case_provider.lock().unwrap();
        provider.suspend("Demo suspension for cache management".to_string())?;
    }
    
    std::thread::sleep(Duration::from_millis(100));
    
    // Reactivate provider
    {
        let mut provider = test_case_provider.lock().unwrap();
        provider.reactivate()?;
    }
    
    println!("\n♻️  Demo Step 4: Testing cache reuse");
    println!("-----------------------------------");
    
    // Request same provider again - should reuse
    let reused_provider = manager.get_provider(
        "demo_random",
        LifecycleScope::TestCase,
        Some("reuse_test_context".to_string()),
        None,
    )?;
    
    // Verify it's the same instance
    {
        let original = test_case_provider.lock().unwrap();
        let reused = reused_provider.lock().unwrap();
        println!("🔍 Original provider ID: {}", original.metadata().provider_id);
        println!("🔍 Reused provider ID: {}", reused.metadata().provider_id);
        println!("♻️  Cache reuse: {}", original.metadata().provider_id == reused.metadata().provider_id);
        println!("📊 Reused provider access count: {}", reused.metadata().access_count);
    }
    
    println!("\n🧹 Demo Step 5: Testing cleanup tasks");
    println!("-------------------------------------");
    
    // Run cleanup tasks
    manager.run_cleanup_tasks()?;
    
    println!("\n📊 Demo Step 6: Manager statistics");
    println!("----------------------------------");
    
    let stats = manager.get_statistics();
    println!("📈 Manager statistics:");
    for (key, value) in &stats {
        println!("   {}: {:?}", key, value);
    }
    
    println!("\n📝 Demo Step 7: Lifecycle events summary");
    println!("----------------------------------------");
    
    let event_count = hooks.lock().unwrap().events.len();
    println!("📋 Total lifecycle events recorded: {}", event_count);
    
    let hook_events = hooks.lock().unwrap().events.clone();
    for (i, event) in hook_events.iter().enumerate() {
        match event {
            LifecycleEvent::Created { provider_id, scope, .. } => {
                println!("  {}. Created provider '{}' (scope: {:?})", i + 1, provider_id, scope);
            },
            LifecycleEvent::Activated { provider_id, test_context } => {
                println!("  {}. Activated provider '{}' (context: {})", i + 1, provider_id, test_context);
            },
            LifecycleEvent::Suspended { provider_id, reason } => {
                println!("  {}. Suspended provider '{}' (reason: {})", i + 1, provider_id, reason);
            },
            LifecycleEvent::Reactivated { provider_id, cache_duration } => {
                println!("  {}. Reactivated provider '{}' (cached: {:?})", i + 1, provider_id, cache_duration);
            },
            LifecycleEvent::Destroyed { provider_id, scope, total_lifetime } => {
                println!("  {}. Destroyed provider '{}' (scope: {:?}, lifetime: {:?})", i + 1, provider_id, scope, total_lifetime);
            },
            _ => {},
        }
    }
    
    println!("\n🛑 Demo Step 8: Manager shutdown");
    println!("--------------------------------");
    
    // Shutdown manager (cleans up all providers)
    manager.shutdown()?;
    
    // Show final event count
    let final_event_count = hooks.lock().unwrap().events.len();
    println!("🏁 Final lifecycle events recorded: {}", final_event_count);
    println!("📊 Events added during shutdown: {}", final_event_count - event_count);
    
    println!("\n🎉 Provider Lifecycle Management Demo Complete!");
    println!("===============================================");
    println!("✅ Successfully demonstrated:");
    println!("   • Per-test-case and per-test-function lifetime controls");
    println!("   • Context management with setup/teardown hooks");
    println!("   • Provider instance caching and reuse strategies");
    println!("   • Advanced observability and debugging support");
    
    Ok(())
}