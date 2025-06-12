//! Provider Lifecycle Management System
//! 
//! This module implements comprehensive provider lifecycle management with:
//! - Per-test-case and per-test-function lifetime controls
//! - Context management with setup/teardown hooks
//! - Provider instance caching and reuse strategies
//! - Advanced observability and debugging support

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, Duration};
use std::any::{Any, TypeId};
use crate::providers::{
    PrimitiveProvider, ProviderLifetime, ProviderError, ProviderScope, TestCaseContext,
    BackendCapabilities, ProviderFactory, ObservationMessage, TestCaseObservation
};
use crate::choice::{ChoiceValue, Constraints, ChoiceType, IntegerConstraints, FloatConstraints, BooleanConstraints};
use serde_json;

/// Comprehensive lifecycle management scope definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LifecycleScope {
    /// Provider instance created for each test case
    TestCase,
    /// Provider instance shared across test cases in a function
    TestFunction,
    /// Provider instance shared across a test run/session
    TestRun,
    /// Provider instance persists for the entire session
    Session,
    /// Custom scope with specified duration
    Custom(u64),
}

impl From<ProviderLifetime> for LifecycleScope {
    fn from(lifetime: ProviderLifetime) -> Self {
        match lifetime {
            ProviderLifetime::TestCase => LifecycleScope::TestCase,
            ProviderLifetime::TestFunction => LifecycleScope::TestFunction,
            ProviderLifetime::TestRun => LifecycleScope::TestRun,
            ProviderLifetime::Session => LifecycleScope::Session,
        }
    }
}

/// Lifecycle event types for provider management
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleEvent {
    /// Provider instance creation
    Created {
        provider_id: String,
        scope: LifecycleScope,
        timestamp: SystemTime,
    },
    /// Provider initialization (setup)
    Initialized {
        provider_id: String,
        context_data: HashMap<String, serde_json::Value>,
    },
    /// Provider entering active use
    Activated {
        provider_id: String,
        test_context: String,
    },
    /// Provider being suspended/cached
    Suspended {
        provider_id: String,
        reason: String,
    },
    /// Provider reactivation from cache
    Reactivated {
        provider_id: String,
        cache_duration: Duration,
    },
    /// Provider cleanup (teardown)
    Cleanup {
        provider_id: String,
        success: bool,
        cleanup_data: HashMap<String, serde_json::Value>,
    },
    /// Provider destruction/disposal
    Destroyed {
        provider_id: String,
        scope: LifecycleScope,
        total_lifetime: Duration,
    },
    /// Error in lifecycle management
    Error {
        provider_id: String,
        error: String,
        context: String,
    },
}

/// Lifecycle hook definitions for provider management
pub trait LifecycleHooks: Send + Sync {
    /// Called when a provider is created
    fn on_provider_created(&mut self, provider_id: &str, scope: LifecycleScope) -> Result<(), ProviderError> {
        println!("LIFECYCLE_HOOKS DEBUG: Provider created: {} (scope: {:?})", provider_id, scope);
        Ok(())
    }
    
    /// Called before provider setup/initialization
    fn before_setup(&mut self, provider_id: &str, context: &HashMap<String, serde_json::Value>) -> Result<(), ProviderError> {
        println!("LIFECYCLE_HOOKS DEBUG: Before setup: {} (context keys: {:?})", provider_id, context.keys().collect::<Vec<_>>());
        Ok(())
    }
    
    /// Called after provider setup/initialization
    fn after_setup(&mut self, provider_id: &str, success: bool) -> Result<(), ProviderError> {
        println!("LIFECYCLE_HOOKS DEBUG: After setup: {} (success: {})", provider_id, success);
        Ok(())
    }
    
    /// Called when provider is activated for use
    fn on_provider_activated(&mut self, provider_id: &str, test_context: &str) -> Result<(), ProviderError> {
        println!("LIFECYCLE_HOOKS DEBUG: Provider activated: {} in context: {}", provider_id, test_context);
        Ok(())
    }
    
    /// Called when provider is suspended/cached
    fn on_provider_suspended(&mut self, provider_id: &str, reason: &str) -> Result<(), ProviderError> {
        println!("LIFECYCLE_HOOKS DEBUG: Provider suspended: {} (reason: {})", provider_id, reason);
        Ok(())
    }
    
    /// Called when provider is reactivated from cache
    fn on_provider_reactivated(&mut self, provider_id: &str, cache_duration: Duration) -> Result<(), ProviderError> {
        println!("LIFECYCLE_HOOKS DEBUG: Provider reactivated: {} (cached for: {:?})", provider_id, cache_duration);
        Ok(())
    }
    
    /// Called before provider cleanup/teardown
    fn before_cleanup(&mut self, provider_id: &str, context: &HashMap<String, serde_json::Value>) -> Result<(), ProviderError> {
        println!("LIFECYCLE_HOOKS DEBUG: Before cleanup: {} (context keys: {:?})", provider_id, context.keys().collect::<Vec<_>>());
        Ok(())
    }
    
    /// Called after provider cleanup/teardown
    fn after_cleanup(&mut self, provider_id: &str, success: bool, cleanup_data: &HashMap<String, serde_json::Value>) -> Result<(), ProviderError> {
        println!("LIFECYCLE_HOOKS DEBUG: After cleanup: {} (success: {}, data keys: {:?})", 
                provider_id, success, cleanup_data.keys().collect::<Vec<_>>());
        Ok(())
    }
    
    /// Called when provider is destroyed
    fn on_provider_destroyed(&mut self, provider_id: &str, scope: LifecycleScope, total_lifetime: Duration) -> Result<(), ProviderError> {
        println!("LIFECYCLE_HOOKS DEBUG: Provider destroyed: {} (scope: {:?}, lifetime: {:?})", 
                provider_id, scope, total_lifetime);
        Ok(())
    }
    
    /// Called on lifecycle errors
    fn on_lifecycle_error(&mut self, provider_id: &str, error: &str, context: &str) -> Result<(), ProviderError> {
        println!("LIFECYCLE_HOOKS ERROR: Provider {} error: {} (context: {})", provider_id, error, context);
        Ok(())
    }
}

/// Default implementation of lifecycle hooks
#[derive(Debug)]
pub struct DefaultLifecycleHooks {
    pub events: Vec<LifecycleEvent>,
    pub verbose: bool,
}

impl DefaultLifecycleHooks {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            verbose: true,
        }
    }
    
    pub fn with_verbose(verbose: bool) -> Self {
        Self {
            events: Vec::new(),
            verbose,
        }
    }
    
    fn log_event(&mut self, event: LifecycleEvent) {
        if self.verbose {
            println!("LIFECYCLE_EVENT: {:?}", event);
        }
        self.events.push(event);
    }
}

impl LifecycleHooks for DefaultLifecycleHooks {
    fn on_provider_created(&mut self, provider_id: &str, scope: LifecycleScope) -> Result<(), ProviderError> {
        self.log_event(LifecycleEvent::Created {
            provider_id: provider_id.to_string(),
            scope,
            timestamp: SystemTime::now(),
        });
        Ok(())
    }
    
    fn after_setup(&mut self, provider_id: &str, success: bool) -> Result<(), ProviderError> {
        if success {
            self.log_event(LifecycleEvent::Initialized {
                provider_id: provider_id.to_string(),
                context_data: HashMap::new(),
            });
        }
        Ok(())
    }
    
    fn on_provider_activated(&mut self, provider_id: &str, test_context: &str) -> Result<(), ProviderError> {
        self.log_event(LifecycleEvent::Activated {
            provider_id: provider_id.to_string(),
            test_context: test_context.to_string(),
        });
        Ok(())
    }
    
    fn on_provider_suspended(&mut self, provider_id: &str, reason: &str) -> Result<(), ProviderError> {
        self.log_event(LifecycleEvent::Suspended {
            provider_id: provider_id.to_string(),
            reason: reason.to_string(),
        });
        Ok(())
    }
    
    fn on_provider_reactivated(&mut self, provider_id: &str, cache_duration: Duration) -> Result<(), ProviderError> {
        self.log_event(LifecycleEvent::Reactivated {
            provider_id: provider_id.to_string(),
            cache_duration,
        });
        Ok(())
    }
    
    fn after_cleanup(&mut self, provider_id: &str, success: bool, cleanup_data: &HashMap<String, serde_json::Value>) -> Result<(), ProviderError> {
        self.log_event(LifecycleEvent::Cleanup {
            provider_id: provider_id.to_string(),
            success,
            cleanup_data: cleanup_data.clone(),
        });
        Ok(())
    }
    
    fn on_provider_destroyed(&mut self, provider_id: &str, scope: LifecycleScope, total_lifetime: Duration) -> Result<(), ProviderError> {
        self.log_event(LifecycleEvent::Destroyed {
            provider_id: provider_id.to_string(),
            scope,
            total_lifetime,
        });
        Ok(())
    }
    
    fn on_lifecycle_error(&mut self, provider_id: &str, error: &str, context: &str) -> Result<(), ProviderError> {
        self.log_event(LifecycleEvent::Error {
            provider_id: provider_id.to_string(),
            error: error.to_string(),
            context: context.to_string(),
        });
        Ok(())
    }
}

/// Provider instance metadata for lifecycle management
#[derive(Debug, Clone)]
pub struct ProviderInstanceMetadata {
    pub provider_id: String,
    pub scope: LifecycleScope,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub access_count: u64,
    pub test_context: Option<String>,
    pub state: ProviderState,
    pub metrics: ProviderMetrics,
    pub custom_data: HashMap<String, serde_json::Value>,
}

/// Provider state tracking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderState {
    Created,
    Initializing,
    Active,
    Suspended,
    Cleanup,
    Destroyed,
    Error(String),
}

/// Provider performance and usage metrics
#[derive(Debug, Clone)]
pub struct ProviderMetrics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub average_operation_time: Duration,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub memory_usage: usize,
    pub custom_metrics: HashMap<String, f64>,
}

impl Default for ProviderMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            average_operation_time: Duration::ZERO,
            cache_hits: 0,
            cache_misses: 0,
            memory_usage: 0,
            custom_metrics: HashMap::new(),
        }
    }
}

/// Managed provider wrapper with lifecycle support
pub struct ManagedProvider {
    provider: Box<dyn PrimitiveProvider>,
    metadata: ProviderInstanceMetadata,
    hooks: Option<Arc<Mutex<dyn LifecycleHooks>>>,
    context: Option<Box<dyn TestCaseContext>>,
}

impl std::fmt::Debug for ManagedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedProvider")
            .field("metadata", &self.metadata)
            .field("has_hooks", &self.hooks.is_some())
            .field("has_context", &self.context.is_some())
            .finish()
    }
}

impl ManagedProvider {
    pub fn new(
        provider: Box<dyn PrimitiveProvider>,
        provider_id: String,
        scope: LifecycleScope,
        hooks: Option<Arc<Mutex<dyn LifecycleHooks>>>,
    ) -> Self {
        let now = SystemTime::now();
        let metadata = ProviderInstanceMetadata {
            provider_id: provider_id.clone(),
            scope,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            test_context: None,
            state: ProviderState::Created,
            metrics: ProviderMetrics::default(),
            custom_data: HashMap::new(),
        };
        
        let mut managed = Self {
            provider,
            metadata,
            hooks: hooks.clone(),
            context: None,
        };
        
        // Trigger creation hook
        if let Some(ref hooks) = hooks {
            if let Ok(mut h) = hooks.lock() {
                let _ = h.on_provider_created(&provider_id, scope);
            }
        }
        
        managed
    }
    
    /// Initialize the provider with setup hooks
    pub fn initialize(&mut self, context_data: HashMap<String, serde_json::Value>) -> Result<(), ProviderError> {
        self.metadata.state = ProviderState::Initializing;
        
        // Call before_setup hook
        if let Some(ref hooks) = self.hooks {
            if let Ok(mut h) = hooks.lock() {
                h.before_setup(&self.metadata.provider_id, &context_data)?;
            }
        }
        
        // Initialize provider context
        self.context = Some(self.provider.per_test_case_context());
        
        // Store context data
        self.metadata.custom_data.extend(context_data);
        self.metadata.state = ProviderState::Active;
        
        // Call after_setup hook
        if let Some(ref hooks) = self.hooks {
            if let Ok(mut h) = hooks.lock() {
                h.after_setup(&self.metadata.provider_id, true)?;
            }
        }
        
        println!("MANAGED_PROVIDER DEBUG: Initialized provider {}", self.metadata.provider_id);
        Ok(())
    }
    
    /// Activate provider for a specific test context
    pub fn activate(&mut self, test_context: String) -> Result<(), ProviderError> {
        self.metadata.last_accessed = SystemTime::now();
        self.metadata.access_count += 1;
        self.metadata.test_context = Some(test_context.clone());
        self.metadata.state = ProviderState::Active;
        
        if let Some(ref mut context) = self.context {
            context.enter_test_case();
        }
        
        // Call activation hook
        if let Some(ref hooks) = self.hooks {
            if let Ok(mut h) = hooks.lock() {
                h.on_provider_activated(&self.metadata.provider_id, &test_context)?;
            }
        }
        
        Ok(())
    }
    
    /// Suspend provider to cache
    pub fn suspend(&mut self, reason: String) -> Result<(), ProviderError> {
        self.metadata.state = ProviderState::Suspended;
        
        if let Some(ref mut context) = self.context {
            context.exit_test_case(true);
        }
        
        // Call suspension hook
        if let Some(ref hooks) = self.hooks {
            if let Ok(mut h) = hooks.lock() {
                h.on_provider_suspended(&self.metadata.provider_id, &reason)?;
            }
        }
        
        Ok(())
    }
    
    /// Reactivate provider from cache
    pub fn reactivate(&mut self) -> Result<(), ProviderError> {
        let cache_duration = self.metadata.last_accessed.elapsed().unwrap_or_default();
        self.metadata.last_accessed = SystemTime::now();
        self.metadata.access_count += 1;
        self.metadata.state = ProviderState::Active;
        self.metadata.metrics.cache_hits += 1;
        
        // Call reactivation hook
        if let Some(ref hooks) = self.hooks {
            if let Ok(mut h) = hooks.lock() {
                h.on_provider_reactivated(&self.metadata.provider_id, cache_duration)?;
            }
        }
        
        Ok(())
    }
    
    /// Cleanup provider with teardown hooks
    pub fn cleanup(&mut self) -> Result<(), ProviderError> {
        self.metadata.state = ProviderState::Cleanup;
        
        let cleanup_context = self.metadata.custom_data.clone();
        
        // Call before_cleanup hook
        if let Some(ref hooks) = self.hooks {
            if let Ok(mut h) = hooks.lock() {
                h.before_cleanup(&self.metadata.provider_id, &cleanup_context)?;
            }
        }
        
        // Perform cleanup
        let success = if let Some(ref mut context) = self.context {
            context.exit_test_case(true);
            true
        } else {
            true
        };
        
        let cleanup_data = self.get_cleanup_data();
        
        // Call after_cleanup hook
        if let Some(ref hooks) = self.hooks {
            if let Ok(mut h) = hooks.lock() {
                h.after_cleanup(&self.metadata.provider_id, success, &cleanup_data)?;
            }
        }
        
        Ok(())
    }
    
    /// Destroy provider instance
    pub fn destroy(&mut self) -> Result<(), ProviderError> {
        let total_lifetime = self.metadata.created_at.elapsed().unwrap_or_default();
        
        // Cleanup if not already done
        if self.metadata.state != ProviderState::Cleanup {
            self.cleanup()?;
        }
        
        self.metadata.state = ProviderState::Destroyed;
        
        // Call destruction hook
        if let Some(ref hooks) = self.hooks {
            if let Ok(mut h) = hooks.lock() {
                h.on_provider_destroyed(&self.metadata.provider_id, self.metadata.scope, total_lifetime)?;
            }
        }
        
        println!("MANAGED_PROVIDER DEBUG: Destroyed provider {} after {:?}", 
                self.metadata.provider_id, total_lifetime);
        Ok(())
    }
    
    /// Get cleanup data for teardown hooks
    fn get_cleanup_data(&self) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert("total_operations".to_string(), 
                   serde_json::Value::Number(serde_json::Number::from(self.metadata.metrics.total_operations)));
        data.insert("successful_operations".to_string(), 
                   serde_json::Value::Number(serde_json::Number::from(self.metadata.metrics.successful_operations)));
        data.insert("failed_operations".to_string(), 
                   serde_json::Value::Number(serde_json::Number::from(self.metadata.metrics.failed_operations)));
        data.insert("cache_hits".to_string(), 
                   serde_json::Value::Number(serde_json::Number::from(self.metadata.metrics.cache_hits)));
        data.insert("cache_misses".to_string(), 
                   serde_json::Value::Number(serde_json::Number::from(self.metadata.metrics.cache_misses)));
        data.insert("lifetime_seconds".to_string(), 
                   serde_json::Value::Number(serde_json::Number::from(
                       self.metadata.created_at.elapsed().unwrap_or_default().as_secs())));
        data
    }
    
    /// Update operation metrics
    pub fn record_operation(&mut self, success: bool, duration: Duration) {
        self.metadata.metrics.total_operations += 1;
        if success {
            self.metadata.metrics.successful_operations += 1;
        } else {
            self.metadata.metrics.failed_operations += 1;
        }
        
        // Update average operation time
        let total_time = self.metadata.metrics.average_operation_time * (self.metadata.metrics.total_operations - 1) as u32 + duration;
        self.metadata.metrics.average_operation_time = total_time / self.metadata.metrics.total_operations as u32;
    }
    
    /// Get provider metadata
    pub fn metadata(&self) -> &ProviderInstanceMetadata {
        &self.metadata
    }
    
    /// Get mutable provider reference
    pub fn provider_mut(&mut self) -> &mut Box<dyn PrimitiveProvider> {
        self.metadata.last_accessed = SystemTime::now();
        &mut self.provider
    }
    
    /// Get provider reference
    pub fn provider(&self) -> &Box<dyn PrimitiveProvider> {
        &self.provider
    }
}

/// Provider lifecycle manager with caching and reuse strategies
pub struct ProviderLifecycleManager {
    /// Active provider instances by scope and ID
    providers: HashMap<(LifecycleScope, String), Arc<Mutex<ManagedProvider>>>,
    /// Provider factories for creating new instances
    factories: HashMap<String, Arc<dyn ProviderFactory>>,
    /// Global lifecycle hooks
    global_hooks: Arc<Mutex<dyn LifecycleHooks>>,
    /// Caching configuration per scope
    cache_config: HashMap<LifecycleScope, CacheConfiguration>,
    /// Instance counters for unique ID generation
    instance_counters: HashMap<String, u64>,
    /// Cleanup tasks for periodic maintenance
    cleanup_tasks: Vec<CleanupTask>,
}

/// Cache configuration for different lifecycle scopes
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheConfiguration {
    /// Maximum number of cached instances
    pub max_instances: usize,
    /// Time-to-live for cached instances
    pub ttl: Duration,
    /// Enable LRU eviction policy
    pub lru_eviction: bool,
    /// Custom eviction predicate
    pub custom_eviction: Option<String>,
}

impl Default for CacheConfiguration {
    fn default() -> Self {
        Self {
            max_instances: 10,
            ttl: Duration::from_secs(300), // 5 minutes
            lru_eviction: true,
            custom_eviction: None,
        }
    }
}

/// Cleanup task for periodic maintenance
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CleanupTask {
    pub scope: LifecycleScope,
    pub interval: Duration,
    pub last_run: SystemTime,
    pub task_type: CleanupTaskType,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CleanupTaskType {
    ExpiredInstances,
    MetricsCompaction,
    CacheOptimization,
    ResourceCleanup,
}

impl ProviderLifecycleManager {
    /// Create a new lifecycle manager
    pub fn new() -> Self {
        let mut cache_config = HashMap::new();
        
        // Default cache configurations for different scopes
        cache_config.insert(LifecycleScope::TestCase, CacheConfiguration {
            max_instances: 5,
            ttl: Duration::from_secs(60),
            lru_eviction: true,
            custom_eviction: None,
        });
        
        cache_config.insert(LifecycleScope::TestFunction, CacheConfiguration {
            max_instances: 3,
            ttl: Duration::from_secs(300),
            lru_eviction: true,
            custom_eviction: None,
        });
        
        cache_config.insert(LifecycleScope::TestRun, CacheConfiguration {
            max_instances: 2,
            ttl: Duration::from_secs(1800), // 30 minutes
            lru_eviction: false,
            custom_eviction: None,
        });
        
        cache_config.insert(LifecycleScope::Session, CacheConfiguration {
            max_instances: 1,
            ttl: Duration::from_secs(3600), // 1 hour
            lru_eviction: false,
            custom_eviction: None,
        });
        
        Self {
            providers: HashMap::new(),
            factories: HashMap::new(),
            global_hooks: Arc::new(Mutex::new(DefaultLifecycleHooks::new())),
            cache_config,
            instance_counters: HashMap::new(),
            cleanup_tasks: vec![
                CleanupTask {
                    scope: LifecycleScope::TestCase,
                    interval: Duration::from_secs(30),
                    last_run: SystemTime::now(),
                    task_type: CleanupTaskType::ExpiredInstances,
                },
                CleanupTask {
                    scope: LifecycleScope::TestFunction,
                    interval: Duration::from_secs(60),
                    last_run: SystemTime::now(),
                    task_type: CleanupTaskType::CacheOptimization,
                },
            ],
        }
    }
    
    /// Register a provider factory
    pub fn register_factory(&mut self, factory: Arc<dyn ProviderFactory>) {
        let name = factory.name().to_string();
        self.factories.insert(name.clone(), factory);
        self.instance_counters.insert(name.clone(), 0);
        println!("LIFECYCLE_MANAGER DEBUG: Registered factory: {}", name);
    }
    
    /// Set global lifecycle hooks
    pub fn set_global_hooks(&mut self, hooks: Arc<Mutex<dyn LifecycleHooks>>) {
        self.global_hooks = hooks;
    }
    
    /// Configure cache settings for a specific scope
    pub fn configure_cache(&mut self, scope: LifecycleScope, config: CacheConfiguration) {
        self.cache_config.insert(scope, config);
        println!("LIFECYCLE_MANAGER DEBUG: Configured cache for scope {:?}", scope);
    }
    
    /// Get or create a provider instance with lifecycle management
    pub fn get_provider(
        &mut self, 
        provider_name: &str, 
        scope: LifecycleScope,
        test_context: Option<String>,
        context_data: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Arc<Mutex<ManagedProvider>>, ProviderError> {
        
        // Check for existing instance first
        if let Some(existing) = self.find_existing_provider(provider_name, scope) {
            println!("LIFECYCLE_MANAGER DEBUG: Reusing existing provider: {} (scope: {:?})", provider_name, scope);
            
            // Reactivate from cache
            if let Ok(mut provider) = existing.lock() {
                provider.reactivate()?;
                if let Some(ref test_ctx) = test_context {
                    provider.activate(test_ctx.clone())?;
                }
            }
            
            return Ok(existing);
        }
        
        // Create new instance
        let provider_instance = self.create_new_provider(provider_name, scope, context_data)?;
        
        // Activate for test context
        if let Some(test_ctx) = test_context {
            if let Ok(mut provider) = provider_instance.lock() {
                provider.activate(test_ctx)?;
            }
        }
        
        println!("LIFECYCLE_MANAGER DEBUG: Created new provider: {} (scope: {:?})", provider_name, scope);
        Ok(provider_instance)
    }
    
    /// Find existing provider instance in cache
    fn find_existing_provider(&self, provider_name: &str, scope: LifecycleScope) -> Option<Arc<Mutex<ManagedProvider>>> {
        // Look for exact scope match first
        for ((cached_scope, cached_name), provider) in &self.providers {
            if cached_name == provider_name && *cached_scope == scope {
                // Check if provider is still valid
                if let Ok(p) = provider.lock() {
                    if matches!(p.metadata.state, ProviderState::Active | ProviderState::Suspended) {
                        return Some(provider.clone());
                    }
                }
            }
        }
        
        // Look for compatible scope (higher lifetime can serve lower lifetime needs)
        let compatible_scopes = match scope {
            LifecycleScope::TestCase => vec![
                LifecycleScope::TestFunction,
                LifecycleScope::TestRun,
                LifecycleScope::Session,
            ],
            LifecycleScope::TestFunction => vec![
                LifecycleScope::TestRun,
                LifecycleScope::Session,
            ],
            LifecycleScope::TestRun => vec![
                LifecycleScope::Session,
            ],
            _ => vec![],
        };
        
        for compatible_scope in compatible_scopes {
            for ((cached_scope, cached_name), provider) in &self.providers {
                if cached_name == provider_name && *cached_scope == compatible_scope {
                    if let Ok(p) = provider.lock() {
                        if matches!(p.metadata.state, ProviderState::Active | ProviderState::Suspended) {
                            return Some(provider.clone());
                        }
                    }
                }
            }
        }
        
        None
    }
    
    /// Create a new provider instance
    fn create_new_provider(
        &mut self,
        provider_name: &str,
        scope: LifecycleScope,
        context_data: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Arc<Mutex<ManagedProvider>>, ProviderError> {
        
        let factory = self.factories.get(provider_name)
            .ok_or_else(|| ProviderError::ConfigError(format!("Provider factory '{}' not found", provider_name)))?;
        
        // Generate unique provider ID
        let counter = self.instance_counters.get_mut(provider_name).unwrap();
        *counter += 1;
        let provider_id = format!("{}_{}", provider_name, counter);
        
        // Create base provider instance
        let base_provider = factory.create_provider();
        
        // Create managed provider with lifecycle support
        let mut managed_provider = ManagedProvider::new(
            base_provider,
            provider_id.clone(),
            scope,
            Some(self.global_hooks.clone()),
        );
        
        // Initialize with context data
        let init_data = context_data.unwrap_or_default();
        managed_provider.initialize(init_data)?;
        
        let provider_arc = Arc::new(Mutex::new(managed_provider));
        
        // Store in cache
        self.providers.insert((scope, provider_id), provider_arc.clone());
        
        // Apply cache policies
        self.apply_cache_policies(scope)?;
        
        Ok(provider_arc)
    }
    
    /// Apply cache policies for the given scope
    fn apply_cache_policies(&mut self, scope: LifecycleScope) -> Result<(), ProviderError> {
        let config = self.cache_config.get(&scope).cloned().unwrap_or_default();
        
        // Count instances for this scope
        let scope_instances: Vec<_> = self.providers.iter()
            .filter(|((s, _), _)| *s == scope)
            .map(|(key, provider)| (key.clone(), provider.clone()))
            .collect();
        
        if scope_instances.len() > config.max_instances {
            println!("LIFECYCLE_MANAGER DEBUG: Cache limit exceeded for scope {:?}, applying eviction", scope);
            
            if config.lru_eviction {
                self.evict_lru_instances(scope, scope_instances.len() - config.max_instances)?;
            }
        }
        
        Ok(())
    }
    
    /// Evict least recently used instances
    fn evict_lru_instances(&mut self, scope: LifecycleScope, count: usize) -> Result<(), ProviderError> {
        let mut candidates: Vec<_> = self.providers.iter()
            .filter(|((s, _), _)| *s == scope)
            .filter_map(|(key, provider)| {
                if let Ok(p) = provider.lock() {
                    Some((key.clone(), p.metadata.last_accessed))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by last accessed time (oldest first)
        candidates.sort_by_key(|(_, last_accessed)| *last_accessed);
        
        for (key, _) in candidates.into_iter().take(count) {
            if let Some(provider) = self.providers.remove(&key) {
                if let Ok(mut p) = provider.lock() {
                    let _ = p.destroy();
                }
                println!("LIFECYCLE_MANAGER DEBUG: Evicted provider: {:?}", key);
            }
        }
        
        Ok(())
    }
    
    /// Clean up expired instances
    pub fn cleanup_expired_instances(&mut self) -> Result<usize, ProviderError> {
        let mut removed_count = 0;
        let now = SystemTime::now();
        
        let mut to_remove = Vec::new();
        
        for ((scope, _), provider) in &self.providers {
            let config = self.cache_config.get(scope).cloned().unwrap_or_default();
            
            if let Ok(p) = provider.lock() {
                if let Ok(age) = p.metadata.created_at.elapsed() {
                    if age > config.ttl || matches!(p.metadata.state, ProviderState::Error(_)) {
                        to_remove.push((scope.clone(), p.metadata.provider_id.clone()));
                    }
                }
            }
        }
        
        for (scope, provider_id) in to_remove {
            if let Some(provider) = self.providers.remove(&(scope, provider_id.clone())) {
                if let Ok(mut p) = provider.lock() {
                    let _ = p.destroy();
                }
                removed_count += 1;
                println!("LIFECYCLE_MANAGER DEBUG: Cleaned up expired provider: {}", provider_id);
            }
        }
        
        Ok(removed_count)
    }
    
    /// Run periodic cleanup tasks
    pub fn run_cleanup_tasks(&mut self) -> Result<(), ProviderError> {
        let now = SystemTime::now();
        
        // Collect tasks that need to run
        let mut tasks_to_run = Vec::new();
        for (index, task) in self.cleanup_tasks.iter().enumerate() {
            if now.duration_since(task.last_run).unwrap_or_default() >= task.interval {
                tasks_to_run.push((index, task.task_type.clone()));
            }
        }
        
        // Execute tasks
        for (index, task_type) in tasks_to_run {
            match task_type {
                CleanupTaskType::ExpiredInstances => {
                    let removed = self.cleanup_expired_instances()?;
                    println!("LIFECYCLE_MANAGER DEBUG: Cleanup task removed {} expired instances", removed);
                },
                CleanupTaskType::CacheOptimization => {
                    self.optimize_cache()?;
                },
                CleanupTaskType::MetricsCompaction => {
                    self.compact_metrics()?;
                },
                CleanupTaskType::ResourceCleanup => {
                    self.cleanup_resources()?;
                },
            }
            self.cleanup_tasks[index].last_run = now;
        }
        
        Ok(())
    }
    
    /// Optimize cache usage and performance
    fn optimize_cache(&mut self) -> Result<(), ProviderError> {
        println!("LIFECYCLE_MANAGER DEBUG: Running cache optimization");
        
        // Analyze usage patterns and adjust cache configurations
        for (scope, config) in &mut self.cache_config {
            let scope_instances: Vec<_> = self.providers.iter()
                .filter(|((s, _), _)| s == scope)
                .filter_map(|(_, provider)| {
                    provider.lock().ok().map(|p| p.metadata.access_count)
                })
                .collect();
            
            if !scope_instances.is_empty() {
                let avg_access = scope_instances.iter().sum::<u64>() / scope_instances.len() as u64;
                
                // Adjust cache size based on usage
                if avg_access > 10 && config.max_instances < 20 {
                    config.max_instances += 1;
                    println!("LIFECYCLE_MANAGER DEBUG: Increased cache size for scope {:?} to {}", scope, config.max_instances);
                } else if avg_access < 2 && config.max_instances > 1 {
                    config.max_instances = config.max_instances.saturating_sub(1);
                    println!("LIFECYCLE_MANAGER DEBUG: Decreased cache size for scope {:?} to {}", scope, config.max_instances);
                }
            }
        }
        
        Ok(())
    }
    
    /// Compact metrics data
    fn compact_metrics(&mut self) -> Result<(), ProviderError> {
        println!("LIFECYCLE_MANAGER DEBUG: Running metrics compaction");
        
        for (_, provider) in &self.providers {
            if let Ok(mut p) = provider.lock() {
                // Reset custom metrics that are no longer relevant
                let last_accessed = p.metadata.last_accessed;
                p.metadata.metrics.custom_metrics.retain(|key, _| {
                    !key.starts_with("temp_") || last_accessed.elapsed().unwrap_or_default() < Duration::from_secs(300)
                });
            }
        }
        
        Ok(())
    }
    
    /// Clean up system resources
    fn cleanup_resources(&mut self) -> Result<(), ProviderError> {
        println!("LIFECYCLE_MANAGER DEBUG: Running resource cleanup");
        
        // Force garbage collection of destroyed providers
        self.providers.retain(|(_, _), provider| {
            if let Ok(p) = provider.lock() {
                !matches!(p.metadata.state, ProviderState::Destroyed)
            } else {
                false
            }
        });
        
        Ok(())
    }
    
    /// Shutdown all providers and cleanup
    pub fn shutdown(&mut self) -> Result<(), ProviderError> {
        println!("LIFECYCLE_MANAGER DEBUG: Shutting down all providers");
        
        let providers_to_shutdown: Vec<_> = self.providers.drain().collect();
        
        for (_, provider) in providers_to_shutdown {
            if let Ok(mut p) = provider.lock() {
                let _ = p.destroy();
            }
        }
        
        println!("LIFECYCLE_MANAGER DEBUG: Shutdown complete");
        Ok(())
    }
    
    /// Get comprehensive manager statistics
    pub fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        // Provider counts by scope
        let mut scope_counts = HashMap::new();
        for ((scope, _), _) in &self.providers {
            *scope_counts.entry(format!("{:?}", scope)).or_insert(0) += 1;
        }
        stats.insert("providers_by_scope".to_string(), serde_json::to_value(scope_counts).unwrap_or_default());
        
        // Cache configurations
        stats.insert("cache_configurations".to_string(), serde_json::to_value(&self.cache_config).unwrap_or_default());
        
        // Total providers
        stats.insert("total_providers".to_string(), serde_json::Value::Number(serde_json::Number::from(self.providers.len())));
        
        // Registered factories
        stats.insert("registered_factories".to_string(), 
                   serde_json::Value::Array(self.factories.keys().map(|k| serde_json::Value::String(k.clone())).collect()));
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{RandomProvider, HypothesisProvider};
    use std::sync::Arc;
    
    struct TestProviderFactory {
        name: String,
    }
    
    impl ProviderFactory for TestProviderFactory {
        fn create_provider(&self) -> Box<dyn PrimitiveProvider> {
            Box::new(RandomProvider::new())
        }
        
        fn name(&self) -> &str {
            &self.name
        }
    }
    
    #[test]
    fn test_lifecycle_manager_basic_operations() {
        let mut manager = ProviderLifecycleManager::new();
        
        // Register a test factory
        let factory = Arc::new(TestProviderFactory {
            name: "test_provider".to_string(),
        });
        manager.register_factory(factory);
        
        // Test provider creation
        let provider = manager.get_provider(
            "test_provider",
            LifecycleScope::TestCase,
            Some("test_context_1".to_string()),
            None,
        ).unwrap();
        
        // Verify provider is managed
        assert!(provider.lock().is_ok());
        let metadata = provider.lock().unwrap().metadata().clone();
        assert_eq!(metadata.scope, LifecycleScope::TestCase);
        assert_eq!(metadata.state, ProviderState::Active);
        assert!(metadata.test_context.is_some());
        
        // Test provider reuse
        let provider2 = manager.get_provider(
            "test_provider",
            LifecycleScope::TestCase,
            Some("test_context_2".to_string()),
            None,
        ).unwrap();
        
        // Should be same instance (reused)
        let metadata2 = provider2.lock().unwrap().metadata().clone();
        assert_eq!(metadata.provider_id, metadata2.provider_id);
        assert!(metadata2.access_count > metadata.access_count);
    }
    
    #[test]
    fn test_provider_lifecycle_hooks() {
        let hooks = Arc::new(Mutex::new(DefaultLifecycleHooks::new()));
        let mut manager = ProviderLifecycleManager::new();
        manager.set_global_hooks(hooks.clone());
        
        let factory = Arc::new(TestProviderFactory {
            name: "hook_test".to_string(),
        });
        manager.register_factory(factory);
        
        // Create provider - should trigger hooks
        let provider = manager.get_provider(
            "hook_test",
            LifecycleScope::TestFunction,
            Some("hook_test_context".to_string()),
            Some({
                let mut data = HashMap::new();
                data.insert("test_key".to_string(), serde_json::Value::String("test_value".to_string()));
                data
            }),
        ).unwrap();
        
        // Verify hooks were called
        let hook_events = hooks.lock().unwrap().events.clone();
        assert!(!hook_events.is_empty());
        
        // Check for creation event
        let has_creation = hook_events.iter().any(|event| {
            matches!(event, LifecycleEvent::Created { .. })
        });
        assert!(has_creation);
        
        // Check for activation event
        let has_activation = hook_events.iter().any(|event| {
            matches!(event, LifecycleEvent::Activated { .. })
        });
        assert!(has_activation);
        
        // Test suspension
        {
            let mut p = provider.lock().unwrap();
            p.suspend("test suspension".to_string()).unwrap();
        }
        
        // Check for suspension event
        let hook_events = hooks.lock().unwrap().events.clone();
        let has_suspension = hook_events.iter().any(|event| {
            matches!(event, LifecycleEvent::Suspended { .. })
        });
        assert!(has_suspension);
    }
    
    #[test]
    fn test_cache_configuration_and_eviction() {
        let mut manager = ProviderLifecycleManager::new();
        
        // Configure small cache for testing
        manager.configure_cache(LifecycleScope::TestCase, CacheConfiguration {
            max_instances: 2,
            ttl: Duration::from_secs(1),
            lru_eviction: true,
            custom_eviction: None,
        });
        
        let factory = Arc::new(TestProviderFactory {
            name: "cache_test".to_string(),
        });
        manager.register_factory(factory);
        
        // Create multiple providers to test eviction
        let _provider1 = manager.get_provider(
            "cache_test",
            LifecycleScope::TestCase,
            Some("context_1".to_string()),
            None,
        ).unwrap();
        
        std::thread::sleep(Duration::from_millis(100));
        
        let _provider2 = manager.get_provider(
            "cache_test",
            LifecycleScope::TestCase,
            Some("context_2".to_string()),
            None,
        ).unwrap();
        
        std::thread::sleep(Duration::from_millis(100));
        
        // This should trigger eviction
        let _provider3 = manager.get_provider(
            "cache_test",
            LifecycleScope::TestCase,
            Some("context_3".to_string()),
            None,
        ).unwrap();
        
        // Verify cache size constraint
        let scope_count = manager.providers.iter()
            .filter(|((scope, _), _)| *scope == LifecycleScope::TestCase)
            .count();
        assert!(scope_count <= 2);
    }
    
    #[test]
    fn test_cleanup_expired_instances() {
        let mut manager = ProviderLifecycleManager::new();
        
        // Configure very short TTL for testing
        manager.configure_cache(LifecycleScope::TestCase, CacheConfiguration {
            max_instances: 10,
            ttl: Duration::from_millis(50),
            lru_eviction: true,
            custom_eviction: None,
        });
        
        let factory = Arc::new(TestProviderFactory {
            name: "expiry_test".to_string(),
        });
        manager.register_factory(factory);
        
        // Create provider
        let _provider = manager.get_provider(
            "expiry_test",
            LifecycleScope::TestCase,
            Some("expiry_context".to_string()),
            None,
        ).unwrap();
        
        assert_eq!(manager.providers.len(), 1);
        
        // Wait for expiry
        std::thread::sleep(Duration::from_millis(100));
        
        // Run cleanup
        let removed_count = manager.cleanup_expired_instances().unwrap();
        assert_eq!(removed_count, 1);
        assert_eq!(manager.providers.len(), 0);
    }
    
    #[test]
    fn test_managed_provider_operations() {
        let hooks = Arc::new(Mutex::new(DefaultLifecycleHooks::new()));
        let base_provider = Box::new(RandomProvider::new());
        let mut managed = ManagedProvider::new(
            base_provider,
            "test_provider_123".to_string(),
            LifecycleScope::TestFunction,
            Some(hooks.clone()),
        );
        
        // Test initialization
        let mut context_data = HashMap::new();
        context_data.insert("init_param".to_string(), serde_json::Value::String("init_value".to_string()));
        managed.initialize(context_data).unwrap();
        
        assert_eq!(managed.metadata.state, ProviderState::Active);
        assert!(managed.metadata.custom_data.contains_key("init_param"));
        
        // Test activation
        managed.activate("test_activation_context".to_string()).unwrap();
        assert_eq!(managed.metadata.test_context, Some("test_activation_context".to_string()));
        assert!(managed.metadata.access_count > 0);
        
        // Test suspension
        managed.suspend("testing suspension".to_string()).unwrap();
        assert_eq!(managed.metadata.state, ProviderState::Suspended);
        
        // Test reactivation
        let initial_access_count = managed.metadata.access_count;
        managed.reactivate().unwrap();
        assert_eq!(managed.metadata.state, ProviderState::Active);
        assert!(managed.metadata.access_count > initial_access_count);
        assert!(managed.metadata.metrics.cache_hits > 0);
        
        // Test operation recording
        managed.record_operation(true, Duration::from_millis(50));
        assert_eq!(managed.metadata.metrics.total_operations, 1);
        assert_eq!(managed.metadata.metrics.successful_operations, 1);
        assert_eq!(managed.metadata.metrics.failed_operations, 0);
        
        managed.record_operation(false, Duration::from_millis(100));
        assert_eq!(managed.metadata.metrics.total_operations, 2);
        assert_eq!(managed.metadata.metrics.successful_operations, 1);
        assert_eq!(managed.metadata.metrics.failed_operations, 1);
        
        // Test cleanup and destruction
        managed.cleanup().unwrap();
        managed.destroy().unwrap();
        assert_eq!(managed.metadata.state, ProviderState::Destroyed);
        
        // Verify hooks were called
        let hook_events = hooks.lock().unwrap().events.clone();
        assert!(hook_events.len() >= 5); // Created, Initialized, Activated, Suspended, Destroyed
    }
    
    #[test]
    fn test_scope_compatibility() {
        let mut manager = ProviderLifecycleManager::new();
        
        let factory = Arc::new(TestProviderFactory {
            name: "scope_test".to_string(),
        });
        manager.register_factory(factory);
        
        // Create session-scoped provider
        let session_provider = manager.get_provider(
            "scope_test",
            LifecycleScope::Session,
            Some("session_context".to_string()),
            None,
        ).unwrap();
        
        let session_id = session_provider.lock().unwrap().metadata.provider_id.clone();
        
        // Request test-case scoped provider - should reuse session provider
        let test_case_provider = manager.get_provider(
            "scope_test",
            LifecycleScope::TestCase,
            Some("test_case_context".to_string()),
            None,
        ).unwrap();
        
        let test_case_id = test_case_provider.lock().unwrap().metadata.provider_id.clone();
        
        // Should be the same provider instance
        assert_eq!(session_id, test_case_id);
        
        // Verify access count increased (reused)
        let access_count = test_case_provider.lock().unwrap().metadata.access_count;
        assert!(access_count > 1);
    }
    
    #[test]
    fn test_periodic_cleanup_tasks() {
        let mut manager = ProviderLifecycleManager::new();
        
        // Add custom cleanup task with very short interval
        manager.cleanup_tasks.push(CleanupTask {
            scope: LifecycleScope::TestCase,
            interval: Duration::from_millis(10),
            last_run: SystemTime::now() - Duration::from_millis(20),
            task_type: CleanupTaskType::ExpiredInstances,
        });
        
        let factory = Arc::new(TestProviderFactory {
            name: "cleanup_task_test".to_string(),
        });
        manager.register_factory(factory);
        
        // Create provider with short TTL
        manager.configure_cache(LifecycleScope::TestCase, CacheConfiguration {
            max_instances: 10,
            ttl: Duration::from_millis(1),
            lru_eviction: true,
            custom_eviction: None,
        });
        
        let _provider = manager.get_provider(
            "cleanup_task_test",
            LifecycleScope::TestCase,
            Some("cleanup_context".to_string()),
            None,
        ).unwrap();
        
        assert_eq!(manager.providers.len(), 1);
        
        // Wait for expiry
        std::thread::sleep(Duration::from_millis(5));
        
        // Run cleanup tasks
        manager.run_cleanup_tasks().unwrap();
        
        // Provider should be cleaned up
        assert_eq!(manager.providers.len(), 0);
    }
    
    #[test]
    fn test_manager_statistics() {
        let mut manager = ProviderLifecycleManager::new();
        
        let factory = Arc::new(TestProviderFactory {
            name: "stats_test".to_string(),
        });
        manager.register_factory(factory);
        
        // Create providers in different scopes
        let _provider1 = manager.get_provider(
            "stats_test",
            LifecycleScope::TestCase,
            Some("stats_context_1".to_string()),
            None,
        ).unwrap();
        
        let _provider2 = manager.get_provider(
            "stats_test",
            LifecycleScope::TestFunction,
            Some("stats_context_2".to_string()),
            None,
        ).unwrap();
        
        let stats = manager.get_statistics();
        
        assert!(stats.contains_key("providers_by_scope"));
        assert!(stats.contains_key("cache_configurations"));
        assert!(stats.contains_key("total_providers"));
        assert!(stats.contains_key("registered_factories"));
        
        // Verify total count
        if let Some(total) = stats.get("total_providers") {
            if let Some(count) = total.as_u64() {
                assert_eq!(count, 2);
            }
        }
        
        // Verify factory registration
        if let Some(factories) = stats.get("registered_factories") {
            if let Some(array) = factories.as_array() {
                assert!(array.iter().any(|v| v.as_str() == Some("stats_test")));
            }
        }
    }
    
    #[test]
    fn test_manager_shutdown() {
        let hooks = Arc::new(Mutex::new(DefaultLifecycleHooks::new()));
        let mut manager = ProviderLifecycleManager::new();
        manager.set_global_hooks(hooks.clone());
        
        let factory = Arc::new(TestProviderFactory {
            name: "shutdown_test".to_string(),
        });
        manager.register_factory(factory);
        
        // Create multiple providers
        let _provider1 = manager.get_provider(
            "shutdown_test",
            LifecycleScope::TestCase,
            Some("shutdown_context_1".to_string()),
            None,
        ).unwrap();
        
        let _provider2 = manager.get_provider(
            "shutdown_test",
            LifecycleScope::TestFunction,
            Some("shutdown_context_2".to_string()),
            None,
        ).unwrap();
        
        assert_eq!(manager.providers.len(), 2);
        
        // Shutdown manager
        manager.shutdown().unwrap();
        
        // All providers should be removed
        assert_eq!(manager.providers.len(), 0);
        
        // Verify destruction events
        let hook_events = hooks.lock().unwrap().events.clone();
        let destruction_events = hook_events.iter()
            .filter(|event| matches!(event, LifecycleEvent::Destroyed { .. }))
            .count();
        assert_eq!(destruction_events, 2);
    }
}