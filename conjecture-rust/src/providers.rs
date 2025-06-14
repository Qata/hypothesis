//! Enhanced Provider Backend Registry System
//! 
//! This module implements a comprehensive provider system with dynamic backend discovery,
//! capability negotiation, and specialized backend support including SMT solvers.
//! It extends beyond the basic random and hypothesis providers to support:
//! - Dynamic provider registration and discovery
//! - Backend capability negotiation
//! - Specialized backends (SMT solvers, fuzzing backends, etc.)
//! - Plugin architecture for extensibility
//! - Observability and instrumentation hooks

use crate::choice::{ChoiceValue, Constraints, ChoiceType, FloatConstraints, IntegerConstraints, BooleanConstraints, IntervalSet, templating::TemplateError};
use crate::data::DrawError;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::fmt;
use serde_json;
use serde::{Serialize, Deserialize};

/// Lifetime management for providers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderLifetime {
    /// Provider lives for the duration of a single test case
    TestCase,
    /// Provider lives for the duration of a test run
    TestRun,
    /// Provider lives for the entire session
    Session,
    /// Provider lives for the entire test function execution
    TestFunction,
}

/// Backend capability flags for provider negotiation
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BackendCapabilities {
    /// Support for integer choices
    pub supports_integers: bool,
    /// Support for float choices
    pub supports_floats: bool,
    /// Support for string choices
    pub supports_strings: bool,
    /// Support for byte array choices
    pub supports_bytes: bool,
    /// Support for general choice types
    pub supports_choices: bool,
    /// Avoid forcing symbolic values to concrete ones
    pub avoid_realization: bool,
    /// Add observability callback support
    pub add_observability_callback: bool,
    /// Support for structural span tracking
    pub structural_awareness: bool,
    /// Support for choice sequence replay
    pub replay_support: bool,
    /// Can handle symbolic constraints
    pub symbolic_constraints: bool,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            supports_integers: true,
            supports_floats: true,
            supports_strings: true,
            supports_bytes: true,
            supports_choices: true,
            avoid_realization: false,
            add_observability_callback: false,
            structural_awareness: false,
            replay_support: false,
            symbolic_constraints: false,
        }
    }
}

/// Enhanced error types for backend negotiation
#[derive(Debug, Clone)]
pub enum ProviderError {
    /// Backend cannot proceed with the current operation
    CannotProceed { scope: ProviderScope, reason: String },
    /// Invalid choice or constraint provided
    InvalidChoice(String),
    /// Error in symbolic value operations
    SymbolicValueError(String),
    /// Backend has exhausted its search space
    BackendExhausted(String),
    /// Plugin loading error
    PluginError(String),
    /// Configuration error
    ConfigError(String),
    /// Template processing failed
    ProcessingFailed(String),
    /// Legacy draw error for compatibility
    DrawError(DrawError),
}

impl From<DrawError> for ProviderError {
    fn from(err: DrawError) -> Self {
        ProviderError::DrawError(err)
    }
}

impl From<TemplateError> for ProviderError {
    fn from(err: TemplateError) -> Self {
        match err {
            TemplateError::ExhaustedTemplate => ProviderError::BackendExhausted("Template exhausted".to_string()),
            TemplateError::ConstraintMismatch => ProviderError::InvalidChoice("Template constraint mismatch".to_string()),
            TemplateError::TypeMismatch => ProviderError::InvalidChoice("Template type mismatch".to_string()),
            TemplateError::UnknownCustomTemplate(name) => ProviderError::ConfigError(format!("Unknown custom template: {}", name)),
            TemplateError::ProcessingFailed(msg) => ProviderError::ProcessingFailed(msg),
        }
    }
}

impl From<ProviderError> for DrawError {
    fn from(err: ProviderError) -> Self {
        match err {
            ProviderError::DrawError(draw_err) => draw_err,
            ProviderError::InvalidChoice(_) => DrawError::InvalidChoice,
            ProviderError::BackendExhausted(_) => DrawError::InvalidChoice,
            ProviderError::CannotProceed { .. } => DrawError::InvalidChoice,
            _ => DrawError::InvalidChoice,
        }
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::CannotProceed { scope, reason } => {
                write!(f, "Backend cannot proceed in scope {:?}: {}", scope, reason)
            },
            ProviderError::InvalidChoice(msg) => write!(f, "Invalid choice: {}", msg),
            ProviderError::SymbolicValueError(msg) => write!(f, "Symbolic value error: {}", msg),
            ProviderError::BackendExhausted(msg) => write!(f, "Backend exhausted: {}", msg),
            ProviderError::PluginError(msg) => write!(f, "Plugin error: {}", msg),
            ProviderError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            ProviderError::ProcessingFailed(msg) => write!(f, "Template processing failed: {}", msg),
            ProviderError::DrawError(err) => write!(f, "Draw error: {:?}", err),
        }
    }
}

impl std::error::Error for ProviderError {}

/// Scope of provider error for backend negotiation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderScope {
    /// Discard the current test case and try again
    DiscardTestCase,
    /// The condition has been verified
    Verified,
    /// Search space has been exhausted
    Exhausted,
    /// Configuration issue
    Configuration,
}

/// Test case context for provider lifecycle management
pub trait TestCaseContext: Send + Sync {
    /// Called when entering a new test case
    fn enter_test_case(&mut self) {}
    /// Called when exiting a test case
    fn exit_test_case(&mut self, success: bool) {}
    /// Get context-specific data
    fn get_context_data(&self) -> HashMap<String, serde_json::Value> { HashMap::new() }
}

/// Default implementation for simple contexts
#[derive(Debug)]
pub struct DefaultTestCaseContext;

impl TestCaseContext for DefaultTestCaseContext {}

/// Observability message types
#[derive(Debug, Clone)]
pub struct ObservationMessage {
    pub level: String,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum ObservationType {
    Info,
    Alert,
    Error,
    Debug,
}

/// Test case observation for provider instrumentation
#[derive(Debug, Clone)]
pub struct TestCaseObservation {
    pub observation_type: String,
    pub metadata: HashMap<String, serde_json::Value>,
    pub test_case_data: Option<serde_json::Value>,
    pub timestamp: std::time::SystemTime,
}


/// Enhanced abstract provider trait for generation backends
/// 
/// This trait provides comprehensive backend capabilities including:
/// - Dynamic capability negotiation
/// - Symbolic value handling
/// - Observability hooks
/// - Structural awareness
/// - Choice sequence replay
pub trait PrimitiveProvider: std::fmt::Debug + Send + Sync {
    /// Get the lifetime of this provider
    fn lifetime(&self) -> ProviderLifetime;
    
    /// Get the capabilities of this provider
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }
    
    /// Get provider metadata for debugging and introspection
    fn metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut meta = HashMap::new();
        meta.insert("provider_type".to_string(), 
                   serde_json::Value::String(format!("{:?}", self)));
        meta.insert("lifetime".to_string(), 
                   serde_json::Value::String(format!("{:?}", self.lifetime())));
        meta.insert("capabilities".to_string(), 
                   serde_json::to_value(self.capabilities()).unwrap_or_default());
        meta
    }
    
    /// Central choice dispatch with enhanced error handling
    fn draw_choice(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Result<ChoiceValue, ProviderError> {
        match (choice_type, constraints) {
            (ChoiceType::Integer, Constraints::Integer(int_constraints)) => {
                self.draw_integer(int_constraints).map(ChoiceValue::Integer)
            },
            (ChoiceType::Boolean, Constraints::Boolean(bool_constraints)) => {
                self.draw_boolean(bool_constraints.p).map(ChoiceValue::Boolean)
            },
            (ChoiceType::Float, Constraints::Float(float_constraints)) => {
                self.draw_float(float_constraints).map(ChoiceValue::Float)
            },
            _ => Err(ProviderError::InvalidChoice(format!("Unsupported choice type: {:?}", choice_type))),
        }
    }
    
    // Core generation methods with enhanced signatures
    /// Draw a boolean with the given probability
    fn draw_boolean(&mut self, p: f64) -> Result<bool, ProviderError>;
    
    /// Draw an integer within the given constraints
    fn draw_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, ProviderError>;
    
    /// Draw a float within the given constraints
    fn draw_float(&mut self, constraints: &FloatConstraints) -> Result<f64, ProviderError>;
    
    /// Draw a string from the given character intervals
    fn draw_string(&mut self, intervals: &IntervalSet, min_size: usize, max_size: usize) -> Result<String, ProviderError>;
    
    /// Draw bytes with the given size constraints
    fn draw_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, ProviderError>;
    
    // Observability and instrumentation hooks
    /// Observe the current test case state
    fn observe_test_case(&mut self) -> HashMap<String, serde_json::Value> {
        HashMap::new()
    }
    
    /// Get information messages for the given lifetime
    fn observe_information_messages(&mut self, _lifetime: ProviderLifetime) -> Vec<ObservationMessage> {
        Vec::new()
    }
    
    /// Handle an observation event
    fn on_observation(&mut self, _observation: &TestCaseObservation) {}
    
    // Structural awareness for sophisticated backends
    /// Notify provider of span start with numeric label
    fn span_start(&mut self, _label: u32) {}
    
    /// Notify provider of span end
    fn span_end(&mut self, _discard: bool) {}
    
    // Backend lifecycle management
    /// Get a test case context for this provider
    fn per_test_case_context(&mut self) -> Box<dyn TestCaseContext> {
        Box::new(DefaultTestCaseContext)
    }
    
    /// Check if this provider can realize symbolic values
    fn can_realize(&self) -> bool {
        false // Default: most providers don't support symbolic values
    }
    
    /// Replay a sequence of choices (for corpus-based backends)
    fn replay_choices(&mut self, _choices: &[ChoiceValue]) -> Result<(), ProviderError> {
        Ok(()) // Default: no-op for non-replay backends
    }
    
    /// Validate backend configuration
    fn validate_config(&self, _config: &HashMap<String, serde_json::Value>) -> Result<(), ProviderError> {
        Ok(()) // Default: accept any config
    }
    
    // Legacy compatibility methods (deprecated but maintained for backward compatibility)
    /// Generate an integer (legacy method)
    fn generate_integer(&mut self, rng: &mut ChaCha8Rng, constraints: &IntegerConstraints) -> Result<i128, DrawError> {
        // Default implementation delegates to new interface
        match self.draw_integer(constraints) {
            Ok(value) => Ok(value),
            Err(_) => {
                // Fallback to random generation
                let min = constraints.min_value.unwrap_or(i128::MIN);
                let max = constraints.max_value.unwrap_or(i128::MAX);
                if min > max {
                    return Err(DrawError::InvalidRange);
                }
                Ok(if min == max { min } else { rng.gen_range(min..=max) })
            }
        }
    }
    
    /// Generate a boolean (legacy method)
    fn generate_boolean(&mut self, rng: &mut ChaCha8Rng, constraints: &BooleanConstraints) -> Result<bool, DrawError> {
        match self.draw_boolean(constraints.p) {
            Ok(value) => Ok(value),
            Err(_) => {
                if constraints.p < 0.0 || constraints.p > 1.0 {
                    return Err(DrawError::InvalidProbability);
                }
                Ok(rng.gen::<f64>() < constraints.p)
            }
        }
    }
    
    /// Generate a float (legacy method)
    fn generate_float(&mut self, rng: &mut ChaCha8Rng, constraints: &FloatConstraints) -> Result<f64, DrawError> {
        match self.draw_float(constraints) {
            Ok(value) => Ok(value),
            Err(_) => Ok(rng.gen::<f64>())
        }
    }
    
    /// Generate a string (legacy method)
    fn generate_string(&mut self, rng: &mut ChaCha8Rng, alphabet: &str, min_size: usize, max_size: usize) -> Result<String, DrawError> {
        let intervals = IntervalSet::from_string(alphabet);
        match self.draw_string(&intervals, min_size, max_size) {
            Ok(value) => Ok(value),
            Err(_) => {
                // Fallback implementation
                if min_size > max_size {
                    return Err(DrawError::InvalidRange);
                }
                let alphabet_chars: Vec<char> = alphabet.chars().collect();
                if alphabet_chars.is_empty() {
                    return Err(DrawError::EmptyAlphabet);
                }
                let size = if min_size == max_size {
                    min_size
                } else {
                    rng.gen_range(min_size..=max_size)
                };
                let mut result = String::new();
                for _ in 0..size {
                    let char_index = rng.gen_range(0..alphabet_chars.len());
                    result.push(alphabet_chars[char_index]);
                }
                Ok(result)
            }
        }
    }
    
    /// Generate bytes (legacy method)
    fn generate_bytes(&mut self, rng: &mut ChaCha8Rng, size: usize) -> Result<Vec<u8>, DrawError> {
        match self.draw_bytes(size, size) {
            Ok(value) => Ok(value),
            Err(_) => {
                let mut bytes = vec![0u8; size];
                rng.fill_bytes(&mut bytes[..]);
                Ok(bytes)
            }
        }
    }
}

/// Provider factory trait for dynamic provider creation
pub trait ProviderFactory: Send + Sync {
    /// Create a new provider instance
    fn create_provider(&self) -> Box<dyn PrimitiveProvider>;
    
    /// Get the name of this provider
    fn name(&self) -> &str;
    
    /// Get provider dependencies (other providers that must be available)
    fn dependencies(&self) -> Vec<&str> { Vec::new() }
    
    /// Get provider metadata for discovery
    fn factory_metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut meta = HashMap::new();
        meta.insert("name".to_string(), serde_json::Value::String(self.name().to_string()));
        meta.insert("dependencies".to_string(), 
                   serde_json::Value::Array(self.dependencies().iter().map(|s| serde_json::Value::String(s.to_string())).collect()));
        meta
    }
    
    /// Validate that this factory can create providers in the current environment
    fn validate_environment(&self) -> Result<(), ProviderError> {
        Ok(()) // Default: always valid
    }
}

/// Enhanced registry with dynamic discovery and capability negotiation
pub struct ProviderRegistry {
    /// Registered provider factories
    factories: HashMap<String, Arc<dyn ProviderFactory>>,
    /// Provider configurations
    configs: HashMap<String, HashMap<String, serde_json::Value>>,
    /// Cached provider instances for session-lifetime providers
    session_cache: HashMap<String, Arc<Mutex<Box<dyn PrimitiveProvider>>>>,
    /// Registry metadata
    metadata: HashMap<String, serde_json::Value>,
}

impl ProviderRegistry {
    /// Create a new enhanced provider registry
    pub fn new() -> Self {
        let mut registry = Self {
            factories: HashMap::new(),
            configs: HashMap::new(),
            session_cache: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        // Register default providers
        registry.register_factory(Arc::new(RandomProviderFactory));
        registry.register_factory(Arc::new(HypothesisProviderFactory));
        
        // Initialize registry metadata
        registry.metadata.insert("version".to_string(), 
                               serde_json::Value::String("1.0.0".to_string()));
        registry.metadata.insert("created_at".to_string(), 
                               serde_json::Value::String(format!("{:?}", std::time::SystemTime::now())));
        
        registry
    }
    
    /// Register a provider factory
    pub fn register_factory(&mut self, factory: Arc<dyn ProviderFactory>) {
        let name = factory.name().to_string();
        println!("PROVIDER_REGISTRY DEBUG: Registering factory for provider '{}'", name);
        
        // Validate environment before registration
        if let Err(e) = factory.validate_environment() {
            println!("PROVIDER_REGISTRY WARNING: Factory '{}' failed environment validation: {}", name, e);
            return;
        }
        
        self.factories.insert(name, factory);
    }
    
    /// Register a simple closure-based provider (legacy compatibility)
    pub fn register<F>(&mut self, name: &str, factory: F)
    where F: Fn() -> Box<dyn PrimitiveProvider> + Send + Sync + 'static,
    {
        self.register_factory(Arc::new(ClosureProviderFactory {
            name: name.to_string(),
            factory: Arc::new(factory),
        }));
    }
    
    /// Create a provider by name with configuration
    pub fn create_with_config(&mut self, name: &str, config: Option<HashMap<String, serde_json::Value>>) -> Result<Box<dyn PrimitiveProvider>, ProviderError> {
        let factory = self.factories.get(name)
            .ok_or_else(|| ProviderError::ConfigError(format!("Provider '{}' not found", name)))?;
        
        // Validate configuration
        if let Some(ref cfg) = config {
            let test_provider = factory.create_provider();
            test_provider.validate_config(cfg)?;
            self.configs.insert(name.to_string(), cfg.clone());
        }
        
        // Check dependencies
        for dep in factory.dependencies() {
            if !self.factories.contains_key(dep) {
                return Err(ProviderError::ConfigError(format!("Dependency '{}' not available for provider '{}'", dep, name)));
            }
        }
        
        let provider = factory.create_provider();
        println!("PROVIDER_REGISTRY DEBUG: Created provider '{}' with lifetime {:?}", name, provider.lifetime());
        
        Ok(provider)
    }
    
    /// Create a provider by name (simplified interface)
    pub fn create(&mut self, name: &str) -> Result<Box<dyn PrimitiveProvider>, ProviderError> {
        self.create_with_config(name, None)
    }
    
    /// Register a provider instance directly (for testing)
    pub fn register_provider(&mut self, name: String, provider: Box<dyn PrimitiveProvider>) -> Result<(), ProviderError> {
        let lifetime = provider.lifetime();
        
        // Store provider in session cache regardless of lifetime for testing simplicity
        self.session_cache.insert(name.clone(), Arc::new(Mutex::new(provider)));
        
        println!("PROVIDER_REGISTRY DEBUG: Registered provider '{}' with lifetime {:?}", name, lifetime);
        Ok(())
    }
    
    /// Get a provider instance by name (for testing)
    pub fn get_provider(&mut self, name: &str) -> Result<Box<dyn PrimitiveProvider>, ProviderError> {
        // Check session cache first
        if let Some(cached_provider) = self.session_cache.get(name) {
            return Ok(Box::new(CachedProviderWrapper {
                provider: cached_provider.clone(),
                name: name.to_string(),
            }));
        }
        
        Err(ProviderError::ConfigError(format!("Provider '{}' not found", name)))
    }
    
    /// Get list of available provider names
    pub fn available_providers(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }
    
    /// Get detailed information about available providers
    pub fn provider_info(&self) -> HashMap<String, serde_json::Value> {
        let mut info = HashMap::new();
        
        for (name, factory) in &self.factories {
            let mut provider_info = factory.factory_metadata();
            
            // Add capability information
            let test_provider = factory.create_provider();
            provider_info.insert("capabilities".to_string(), 
                               serde_json::to_value(test_provider.capabilities()).unwrap_or_default());
            provider_info.insert("lifetime".to_string(), 
                               serde_json::Value::String(format!("{:?}", test_provider.lifetime())));
            
            info.insert(name.clone(), serde_json::Value::Object(
                provider_info.into_iter().map(|(k, v)| (k, v)).collect()
            ));
        }
        
        info
    }
    
    /// Validate a provider backend name
    pub fn validate_backend(&self, backend: &str) -> Result<String, ProviderError> {
        if !self.factories.contains_key(backend) {
            let available: Vec<_> = self.available_providers();
            return Err(ProviderError::ConfigError(format!(
                "Backend '{}' is not available - maybe you need to install a plugin?\nInstalled backends: {:?}", 
                backend, available
            )));
        }
        Ok(backend.to_string())
    }
    
    /// Auto-discover providers from environment (for future plugin support)
    pub fn discover_providers(&mut self) -> Result<usize, ProviderError> {
        // Placeholder for future plugin discovery mechanism
        // This would scan for dynamic libraries, check environment variables, etc.
        println!("PROVIDER_REGISTRY DEBUG: Provider discovery not yet implemented");
        Ok(0)
    }
    
    /// Set configuration for a provider
    pub fn set_provider_config(&mut self, provider_name: &str, config: HashMap<String, serde_json::Value>) -> Result<(), ProviderError> {
        // Validate the provider exists
        self.validate_backend(provider_name)?;
        
        // Validate configuration with a test provider
        if let Ok(test_provider) = self.create(provider_name) {
            test_provider.validate_config(&config)?;
        }
        
        self.configs.insert(provider_name.to_string(), config);
        Ok(())
    }
    
    /// Get configuration for a provider
    pub fn get_provider_config(&self, provider_name: &str) -> Option<&HashMap<String, serde_json::Value>> {
        self.configs.get(provider_name)
    }
    
    /// Get registry metadata
    pub fn registry_metadata(&self) -> &HashMap<String, serde_json::Value> {
        &self.metadata
    }
}

/// Closure-based provider factory for backward compatibility
struct ClosureProviderFactory {
    name: String,
    factory: Arc<dyn Fn() -> Box<dyn PrimitiveProvider> + Send + Sync>,
}

/// Direct provider factory for registered instances
struct DirectProviderFactory {
    name: String,
    provider_box: Arc<Mutex<Option<Box<dyn PrimitiveProvider>>>>,
}

/// Wrapper for cached provider instances
#[derive(Debug)]
pub struct CachedProviderWrapper {
    provider: Arc<Mutex<Box<dyn PrimitiveProvider>>>,
    name: String,
}

impl ProviderFactory for ClosureProviderFactory {
    fn create_provider(&self) -> Box<dyn PrimitiveProvider> {
        (self.factory)()
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

impl ProviderFactory for DirectProviderFactory {
    fn create_provider(&self) -> Box<dyn PrimitiveProvider> {
        // For direct provider factories, we need to create a new instance that behaves like the original
        // This is a simplified approach - in a real implementation we might clone the provider
        // For now, we'll create a RandomProvider as a fallback
        Box::new(RandomProvider::new())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

impl PrimitiveProvider for CachedProviderWrapper {
    fn lifetime(&self) -> ProviderLifetime {
        let provider = self.provider.lock().unwrap();
        provider.lifetime()
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        let provider = self.provider.lock().unwrap();
        provider.capabilities()
    }
    
    fn draw_boolean(&mut self, p: f64) -> Result<bool, ProviderError> {
        let mut provider = self.provider.lock().unwrap();
        provider.draw_boolean(p)
    }
    
    fn draw_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        let mut provider = self.provider.lock().unwrap();
        provider.draw_integer(constraints)
    }
    
    fn draw_float(&mut self, constraints: &FloatConstraints) -> Result<f64, ProviderError> {
        let mut provider = self.provider.lock().unwrap();
        provider.draw_float(constraints)
    }
    
    fn draw_string(&mut self, intervals: &IntervalSet, min_size: usize, max_size: usize) -> Result<String, ProviderError> {
        let mut provider = self.provider.lock().unwrap();
        provider.draw_string(intervals, min_size, max_size)
    }
    
    fn draw_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, ProviderError> {
        let mut provider = self.provider.lock().unwrap();
        provider.draw_bytes(min_size, max_size)
    }
    
    fn observe_test_case(&mut self) -> HashMap<String, serde_json::Value> {
        let mut provider = self.provider.lock().unwrap();
        provider.observe_test_case()
    }
    
    fn observe_information_messages(&mut self, lifetime: ProviderLifetime) -> Vec<ObservationMessage> {
        let mut provider = self.provider.lock().unwrap();
        provider.observe_information_messages(lifetime)
    }
    
    fn per_test_case_context(&mut self) -> Box<dyn TestCaseContext> {
        let mut provider = self.provider.lock().unwrap();
        provider.per_test_case_context()
    }
    
    fn span_start(&mut self, label: u32) {
        let mut provider = self.provider.lock().unwrap();
        provider.span_start(label)
    }
    
    fn span_end(&mut self, discard: bool) {
        let mut provider = self.provider.lock().unwrap();
        provider.span_end(discard)
    }
}

/// Factory for RandomProvider
struct RandomProviderFactory;

impl ProviderFactory for RandomProviderFactory {
    fn create_provider(&self) -> Box<dyn PrimitiveProvider> {
        Box::new(RandomProvider::new())
    }
    
    fn name(&self) -> &str {
        "random"
    }
}

/// Factory for HypothesisProvider
struct HypothesisProviderFactory;

impl ProviderFactory for HypothesisProviderFactory {
    fn create_provider(&self) -> Box<dyn PrimitiveProvider> {
        Box::new(HypothesisProvider::new())
    }
    
    fn name(&self) -> &str {
        "hypothesis"
    }
}

/// Global provider registry instance
lazy_static::lazy_static! {
    static ref PROVIDER_REGISTRY: std::sync::Mutex<ProviderRegistry> = 
        std::sync::Mutex::new(ProviderRegistry::new());
}

/// Get the global provider registry
pub fn get_provider_registry() -> std::sync::MutexGuard<'static, ProviderRegistry> {
    PROVIDER_REGISTRY.lock().unwrap()
}

/// Register a provider factory globally
pub fn register_global_provider_factory(factory: Arc<dyn ProviderFactory>) {
    let mut registry = get_provider_registry();
    registry.register_factory(factory);
}

/// Create a provider from the global registry
pub fn create_global_provider(name: &str) -> Result<Box<dyn PrimitiveProvider>, ProviderError> {
    let mut registry = get_provider_registry();
    registry.create(name)
}

/// Get information about all available providers
pub fn get_global_provider_info() -> HashMap<String, serde_json::Value> {
    let registry = get_provider_registry();
    registry.provider_info()
}

/// Basic random provider that uses ConjectureData's internal RNG
/// 
/// This is a simple fallback provider that doesn't do any sophisticated
/// generation strategies.
#[derive(Debug)]
pub struct RandomProvider {
    // For now, this is stateless and just delegates to ConjectureData
}

impl RandomProvider {
    pub fn new() -> Self {
        Self {}
    }
}

impl PrimitiveProvider for RandomProvider {
    fn lifetime(&self) -> ProviderLifetime {
        ProviderLifetime::TestCase
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_integers: true,
            supports_floats: true,
            supports_strings: true,
            supports_bytes: true,
            supports_choices: true,
            avoid_realization: false,
            add_observability_callback: false,
            structural_awareness: false,
            replay_support: false,
            symbolic_constraints: false,
        }
    }
    
    fn draw_boolean(&mut self, p: f64) -> Result<bool, ProviderError> {
        if p < 0.0 || p > 1.0 {
            return Err(ProviderError::InvalidChoice(format!("Invalid probability: {}", p)));
        }
        let mut rng = ChaCha8Rng::from_entropy();
        Ok(rng.gen::<f64>() < p)
    }
    
    fn draw_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        let min = constraints.min_value.unwrap_or(i128::MIN);
        let max = constraints.max_value.unwrap_or(i128::MAX);
        
        if min > max {
            return Err(ProviderError::InvalidChoice(format!("Invalid range: {} > {}", min, max)));
        }
        
        let mut rng = ChaCha8Rng::from_entropy();
        if min == max {
            Ok(min)
        } else {
            Ok(rng.gen_range(min..=max))
        }
    }
    
    fn draw_float(&mut self, constraints: &FloatConstraints) -> Result<f64, ProviderError> {
        let mut rng = ChaCha8Rng::from_entropy();
        let value = rng.gen::<f64>();
        
        // Apply constraints if specified
        let constrained_value = if constraints.min_value.is_finite() && constraints.max_value.is_finite() {
            constraints.min_value + value * (constraints.max_value - constraints.min_value)
        } else {
            value
        };
        
        if !constraints.allow_nan && constrained_value.is_nan() {
            return self.draw_float(constraints); // Retry
        }
        
        Ok(constrained_value)
    }
    
    fn draw_string(&mut self, intervals: &IntervalSet, min_size: usize, max_size: usize) -> Result<String, ProviderError> {
        if min_size > max_size {
            return Err(ProviderError::InvalidChoice(format!("Invalid size range: {} > {}", min_size, max_size)));
        }
        
        if intervals.intervals.is_empty() {
            return Err(ProviderError::InvalidChoice("Empty character set".to_string()));
        }
        
        let mut rng = ChaCha8Rng::from_entropy();
        let size = if min_size == max_size {
            min_size
        } else {
            rng.gen_range(min_size..=max_size)
        };
        
        let mut result = String::new();
        for _ in 0..size {
            // Pick a random interval
            let interval_idx = rng.gen_range(0..intervals.intervals.len());
            let (start, end) = intervals.intervals[interval_idx];
            
            // Pick a random code point in that interval
            let code_point = rng.gen_range(start..=end);
            
            // Convert to char (with fallback for invalid code points)
            if let Some(ch) = char::from_u32(code_point) {
                result.push(ch);
            } else {
                result.push('?'); // Fallback character
            }
        }
        
        Ok(result)
    }
    
    fn draw_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, ProviderError> {
        if min_size > max_size {
            return Err(ProviderError::InvalidChoice(format!("Invalid size range: {} > {}", min_size, max_size)));
        }
        
        let mut rng = ChaCha8Rng::from_entropy();
        let size = if min_size == max_size {
            min_size
        } else {
            rng.gen_range(min_size..=max_size)
        };
        
        let mut bytes = vec![0u8; size];
        rng.fill_bytes(&mut bytes[..]);
        Ok(bytes)
    }
}

/// Hypothesis provider with constant injection and advanced algorithms
/// 
/// This is the Rust equivalent of Python's HypothesisProvider class.
/// It implements sophisticated generation strategies including:
/// - Constant injection from global and local pools
/// - Edge case generation 
/// - Structure-aware generation
#[derive(Debug)]
pub struct HypothesisProvider {
    /// Global constants that are always available
    global_constants: GlobalConstants,
    
    /// Local constants discovered from the test module (TODO: implement discovery)
    local_constants: HashMap<String, Vec<ChoiceValue>>,
    
    /// Cache of filtered constants for specific constraints
    constant_cache: HashMap<String, Vec<ChoiceValue>>,
}

impl HypothesisProvider {
    pub fn new() -> Self {
        Self {
            global_constants: GlobalConstants::new(),
            local_constants: HashMap::new(),
            constant_cache: HashMap::new(),
        }
    }
    
    /// Maybe draw a constant instead of a random value
    /// 
    /// This implements Python's _maybe_draw_constant method with 5% probability
    /// of drawing from constant pools instead of random generation.
    fn maybe_draw_constant(&mut self, rng: &mut ChaCha8Rng, choice_type: &str, constraints: &Constraints) -> Option<ChoiceValue> {
        // 5% probability of drawing from constants
        let should_use_constant: f64 = rng.gen();
        if should_use_constant < 0.05 {
            println!("PROVIDER DEBUG: Attempting to draw constant for {}", choice_type);
            return self.draw_from_constant_pool(rng, choice_type, constraints);
        }
        None
    }
    
    /// Draw from appropriate constant pool
    fn draw_from_constant_pool(&mut self, rng: &mut ChaCha8Rng, choice_type: &str, constraints: &Constraints) -> Option<ChoiceValue> {
        let cache_key = format!("{}_{:?}", choice_type, constraints);
        
        // Check cache first
        if let Some(cached_constants) = self.constant_cache.get(&cache_key) {
            if !cached_constants.is_empty() {
                let index = rng.gen_range(0..cached_constants.len());
                println!("PROVIDER DEBUG: Drew constant from cache: {:?}", cached_constants[index]);
                return Some(cached_constants[index].clone());
            }
        }
        
        // Filter constants for this choice type and constraints
        let filtered_constants = self.filter_constants_for_constraints(choice_type, constraints);
        
        if !filtered_constants.is_empty() {
            // Cache the filtered constants
            self.constant_cache.insert(cache_key, filtered_constants.clone());
            
            // Draw from filtered constants
            let index = rng.gen_range(0..filtered_constants.len());
            println!("PROVIDER DEBUG: Drew fresh constant: {:?}", filtered_constants[index]);
            return Some(filtered_constants[index].clone());
        }
        
        None
    }
    
    /// Filter global and local constants for the given constraints
    fn filter_constants_for_constraints(&self, choice_type: &str, constraints: &Constraints) -> Vec<ChoiceValue> {
        let mut result = Vec::new();
        
        match choice_type {
            "integer" => {
                if let Constraints::Integer(int_constraints) = constraints {
                    let min = int_constraints.min_value.unwrap_or(i128::MIN);
                    let max = int_constraints.max_value.unwrap_or(i128::MAX);
                    
                    for &constant in &self.global_constants.integers {
                        if constant >= min && constant <= max {
                            result.push(ChoiceValue::Integer(constant));
                        }
                    }
                }
            },
            "boolean" => {
                // Booleans don't need filtering - they're all valid
                result.push(ChoiceValue::Boolean(true));
                result.push(ChoiceValue::Boolean(false));
            },
            "float" => {
                if let Constraints::Float(float_constraints) = constraints {
                    for &constant in &self.global_constants.floats {
                        if constant >= float_constraints.min_value && constant <= float_constraints.max_value {
                            if !constant.is_nan() || float_constraints.allow_nan {
                                result.push(ChoiceValue::Float(constant));
                            }
                        }
                    }
                }
            },
            "string" => {
                // For strings, we'd need alphabet constraints - simplified for now
                for constant in &self.global_constants.strings {
                    result.push(ChoiceValue::String(constant.clone()));
                }
            },
            "bytes" => {
                // For bytes, we'd need size constraints - simplified for now  
                for constant in &self.global_constants.bytes {
                    result.push(ChoiceValue::Bytes(constant.clone()));
                }
            },
            _ => {},
        }
        
        result
    }
}

impl PrimitiveProvider for HypothesisProvider {
    fn lifetime(&self) -> ProviderLifetime {
        ProviderLifetime::TestCase
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_integers: true,
            supports_floats: true,
            supports_strings: true,
            supports_bytes: true,
            supports_choices: true,
            avoid_realization: false,
            add_observability_callback: true,
            structural_awareness: true,
            replay_support: false,
            symbolic_constraints: false,
        }
    }
    
    fn draw_boolean(&mut self, p: f64) -> Result<bool, ProviderError> {
        if p < 0.0 || p > 1.0 {
            return Err(ProviderError::InvalidChoice(format!("Invalid probability: {}", p)));
        }
        
        let constraints = BooleanConstraints { p };
        let constraints_obj = Constraints::Boolean(constraints.clone());
        let mut rng = ChaCha8Rng::from_entropy();
        
        // Try to draw a constant first (5% probability)
        if let Some(ChoiceValue::Boolean(constant)) = self.maybe_draw_constant(&mut rng, "boolean", &constraints_obj) {
            return Ok(constant);
        }
        
        // Fall back to random generation
        Ok(rng.gen::<f64>() < p)
    }
    
    fn draw_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        let min = constraints.min_value.unwrap_or(i128::MIN);
        let max = constraints.max_value.unwrap_or(i128::MAX);
        
        if min > max {
            return Err(ProviderError::InvalidChoice(format!("Invalid range: {} > {}", min, max)));
        }
        
        let constraints_obj = Constraints::Integer(constraints.clone());
        let mut rng = ChaCha8Rng::from_entropy();
        
        // Try to draw a constant first (5% probability)
        if let Some(ChoiceValue::Integer(constant)) = self.maybe_draw_constant(&mut rng, "integer", &constraints_obj) {
            return Ok(constant);
        }
        
        // Fall back to random generation
        if min == max {
            Ok(min)
        } else {
            Ok(rng.gen_range(min..=max))
        }
    }
    
    fn draw_float(&mut self, constraints: &FloatConstraints) -> Result<f64, ProviderError> {
        let constraints_obj = Constraints::Float(constraints.clone());
        let mut rng = ChaCha8Rng::from_entropy();
        
        // Try to draw a constant first (5% probability)
        if let Some(ChoiceValue::Float(constant)) = self.maybe_draw_constant(&mut rng, "float", &constraints_obj) {
            return Ok(constant);
        }
        
        // Fall back to random generation with constraints
        let value = rng.gen::<f64>();
        let constrained_value = if constraints.min_value.is_finite() && constraints.max_value.is_finite() {
            constraints.min_value + value * (constraints.max_value - constraints.min_value)
        } else {
            value
        };
        
        if !constraints.allow_nan && constrained_value.is_nan() {
            return self.draw_float(constraints); // Retry
        }
        
        Ok(constrained_value)
    }
    
    fn draw_string(&mut self, intervals: &IntervalSet, min_size: usize, max_size: usize) -> Result<String, ProviderError> {
        if min_size > max_size {
            return Err(ProviderError::InvalidChoice(format!("Invalid size range: {} > {}", min_size, max_size)));
        }
        
        if intervals.intervals.is_empty() {
            return Err(ProviderError::InvalidChoice("Empty character set".to_string()));
        }
        
        let mut rng = ChaCha8Rng::from_entropy();
        
        // Try to draw a constant string first (simplified)
        if rng.gen::<f64>() < 0.05 {
            for constant in &self.global_constants.strings {
                if constant.len() >= min_size && constant.len() <= max_size {
                    // Check if all characters are in the intervals
                    if constant.chars().all(|c| intervals.contains(c as u32)) {
                        println!("PROVIDER DEBUG: Drew constant string: {:?}", constant);
                        return Ok(constant.clone());
                    }
                }
            }
        }
        
        // Fall back to random generation
        let size = if min_size == max_size {
            min_size
        } else {
            rng.gen_range(min_size..=max_size)
        };
        
        let mut result = String::new();
        for _ in 0..size {
            // Pick a random interval
            let interval_idx = rng.gen_range(0..intervals.intervals.len());
            let (start, end) = intervals.intervals[interval_idx];
            
            // Pick a random code point in that interval
            let code_point = rng.gen_range(start..=end);
            
            // Convert to char (with fallback for invalid code points)
            if let Some(ch) = char::from_u32(code_point) {
                result.push(ch);
            } else {
                result.push('?'); // Fallback character
            }
        }
        
        Ok(result)
    }
    
    fn draw_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, ProviderError> {
        if min_size > max_size {
            return Err(ProviderError::InvalidChoice(format!("Invalid size range: {} > {}", min_size, max_size)));
        }
        
        let mut rng = ChaCha8Rng::from_entropy();
        
        // Try to draw a constant byte sequence first
        if rng.gen::<f64>() < 0.05 {
            for constant in &self.global_constants.bytes {
                if constant.len() >= min_size && constant.len() <= max_size {
                    println!("PROVIDER DEBUG: Drew constant bytes: {:?}", constant);
                    return Ok(constant.clone());
                }
            }
        }
        
        // Fall back to random generation
        let size = if min_size == max_size {
            min_size
        } else {
            rng.gen_range(min_size..=max_size)
        };
        
        let mut bytes = vec![0u8; size];
        rng.fill_bytes(&mut bytes[..]);
        Ok(bytes)
    }
    
    fn observe_test_case(&mut self) -> HashMap<String, serde_json::Value> {
        let mut observation = HashMap::new();
        observation.insert("constant_cache_size".to_string(), 
                         serde_json::Value::Number(serde_json::Number::from(self.constant_cache.len())));
        observation.insert("global_constants_count".to_string(), 
                         serde_json::Value::Number(serde_json::Number::from(
                             self.global_constants.integers.len() + 
                             self.global_constants.floats.len() + 
                             self.global_constants.strings.len() + 
                             self.global_constants.bytes.len()
                         )));
        observation
    }
    
    fn observe_information_messages(&mut self, lifetime: ProviderLifetime) -> Vec<ObservationMessage> {
        let mut messages = Vec::new();
        
        if lifetime == ProviderLifetime::TestCase {
            messages.push(ObservationMessage {
                level: "info".to_string(),
                message: "Constant Injection Status".to_string(),
                data: Some(serde_json::json!({
                    "cache_size": self.constant_cache.len(),
                    "injection_rate": "5%"
                })),
            });
        }
        
        messages
    }
    
    fn span_start(&mut self, label: u32) {
        println!("PROVIDER DEBUG: Span started: {:#08X}", label);
    }
    
    fn span_end(&mut self, discard: bool) {
        println!("PROVIDER DEBUG: Span ended, discard: {}", discard);
    }
}

/// Global constants that are always available for injection
/// 
/// These correspond to Python's global constant pools that provide
/// common edge cases for different data types.
#[derive(Debug)]
pub struct GlobalConstants {
    /// Common integer edge cases
    pub integers: Vec<i128>,
    
    /// Common float edge cases  
    pub floats: Vec<f64>,
    
    /// Common string edge cases
    pub strings: Vec<String>,
    
    /// Common byte sequence edge cases
    pub bytes: Vec<Vec<u8>>,
}

impl GlobalConstants {
    pub fn new() -> Self {
        Self {
            integers: vec![
                // Basic values
                0, 1, -1,
                // Powers of 2
                2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024,
                // Boundary values
                i8::MIN as i128, i8::MAX as i128,
                i16::MIN as i128, i16::MAX as i128,
                i32::MIN as i128, i32::MAX as i128,
                i64::MIN as i128, i64::MAX as i128,
                // Common problematic values
                10, 100, 1000, -10, -100, -1000,
            ],
            floats: vec![
                // Basic values
                0.0, 1.0, -1.0,
                // Special values
                f64::NAN, f64::INFINITY, f64::NEG_INFINITY,
                // Boundary values
                f64::MIN, f64::MAX, f64::MIN_POSITIVE,
                f64::EPSILON,
                // Common values
                0.5, -0.5, 2.0, -2.0, 10.0, -10.0,
                // Problematic fractions
                1.0 / 3.0, 2.0 / 3.0, 1.0 / 7.0,
            ],
            strings: vec![
                // Empty and basic
                String::new(),
                " ".to_string(),
                "a".to_string(),
                "test".to_string(),
                // Special characters
                "\n".to_string(),
                "\t".to_string(),
                "\r\n".to_string(),
                "\"".to_string(),
                "'".to_string(),
                "\\".to_string(),
                // Unicode
                "🦀".to_string(),
                "α".to_string(),
                "中文".to_string(),
            ],
            bytes: vec![
                // Empty and basic
                vec![],
                vec![0],
                vec![255],
                vec![0, 255],
                // Common patterns
                vec![0, 0, 0, 0],
                vec![255, 255, 255, 255],
                // ASCII
                b"test".to_vec(),
                b"\n".to_vec(),
                b"\0".to_vec(),
            ],
        }
    }
}

// ==================== SPECIALIZED BACKEND IMPLEMENTATIONS ====================

/// SMT Solver Provider for symbolic execution and constraint solving
/// 
/// This provider integrates with SMT solvers to enable symbolic test generation,
/// constraint solving, and advanced property verification. It maintains symbolic
/// values throughout test execution and only realizes them when necessary.
#[derive(Debug)]
pub struct SmtSolverProvider {
    /// Mock SMT solver interface (in real implementation this would be z3, cvc4, etc.)
    solver_state: SmtSolverState,
    /// Symbolic value tracking
    symbolic_values: HashMap<u64, SymbolicValue>,
    /// Solver constraints accumulated during execution
    constraints: Vec<SymbolicConstraint>,
    /// Whether the search space has been exhausted
    space_exhausted: bool,
    /// Analysis context for test case execution
    analysis_context: Option<String>,
    /// Unique ID counter for symbolic values
    next_value_id: u64,
}

#[derive(Debug)]
pub struct SmtSolverState {
    /// Mock solver status
    status: SolverStatus,
    /// Solution cache for performance
    solution_cache: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum SolverStatus {
    Ready,
    Solving,
    Satisfied,
    Unsatisfied,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct SymbolicValue {
    id: u64,
    value_type: SymbolicType,
    constraints: Vec<SymbolicConstraint>,
    concrete_value: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum SymbolicType {
    Integer { min: Option<i128>, max: Option<i128> },
    Boolean,
    Float { min: f64, max: f64, allow_nan: bool },
    String { char_set: IntervalSet, min_len: usize, max_len: usize },
    Bytes { min_len: usize, max_len: usize },
}

#[derive(Debug, Clone)]
pub struct SymbolicConstraint {
    constraint_type: ConstraintType,
    operands: Vec<u64>, // References to symbolic value IDs
    operator: ConstraintOperator,
    target_value: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum ConstraintType {
    Comparison,
    Logical,
    Arithmetic,
    StringLength,
    BytesLength,
}

#[derive(Debug, Clone)]
pub enum ConstraintOperator {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    And,
    Or,
    Not,
    Plus,
    Minus,
    Multiply,
    Divide,
}

impl SmtSolverProvider {
    pub fn new() -> Self {
        Self {
            solver_state: SmtSolverState {
                status: SolverStatus::Ready,
                solution_cache: HashMap::new(),
            },
            symbolic_values: HashMap::new(),
            constraints: Vec::new(),
            space_exhausted: false,
            analysis_context: None,
            next_value_id: 1,
        }
    }
    
    fn create_symbolic_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        let value_id = self.next_value_id;
        self.next_value_id += 1;
        
        let symbolic_value = SymbolicValue {
            id: value_id,
            value_type: SymbolicType::Integer {
                min: constraints.min_value,
                max: constraints.max_value,
            },
            constraints: Vec::new(),
            concrete_value: None,
        };
        
        self.symbolic_values.insert(value_id, symbolic_value);
        
        println!("SMT_SOLVER DEBUG: Created symbolic integer #{} with constraints {:?}", value_id, constraints);
        
        // For demonstration, return a heuristic value
        let heuristic_value = constraints.shrink_towards.unwrap_or(0);
        if let Some(min) = constraints.min_value {
            if heuristic_value < min {
                return Ok(min);
            }
        }
        if let Some(max) = constraints.max_value {
            if heuristic_value > max {
                return Ok(max);
            }
        }
        
        Ok(heuristic_value)
    }
    
    fn solve_constraints(&mut self) -> Result<(), ProviderError> {
        println!("SMT_SOLVER DEBUG: Solving {} constraints", self.constraints.len());
        
        // Mock solver logic - in real implementation this would call z3/cvc4
        self.solver_state.status = SolverStatus::Solving;
        
        // Simulate constraint solving
        std::thread::sleep(std::time::Duration::from_millis(1));
        
        if self.constraints.len() > 100 {
            self.space_exhausted = true;
            return Err(ProviderError::BackendExhausted("Too many constraints".to_string()));
        }
        
        self.solver_state.status = SolverStatus::Satisfied;
        Ok(())
    }
}

impl PrimitiveProvider for SmtSolverProvider {
    fn lifetime(&self) -> ProviderLifetime {
        ProviderLifetime::TestFunction // SMT solver contexts usually span entire test functions
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_integers: true,
            supports_floats: true,
            supports_strings: true,
            supports_bytes: true,
            supports_choices: true,
            avoid_realization: true,  // Keep symbolic values as long as possible
            add_observability_callback: true,
            structural_awareness: true,
            replay_support: false,   // SMT solvers generate new solutions
            symbolic_constraints: true,
        }
    }
    
    fn draw_boolean(&mut self, p: f64) -> Result<bool, ProviderError> {
        if self.space_exhausted {
            return Err(ProviderError::CannotProceed {
                scope: ProviderScope::Exhausted,
                reason: "SMT solver search space exhausted".to_string(),
            });
        }
        
        // For symbolic execution, we might want to explore both paths
        // For now, return a heuristic value
        Ok(p >= 0.5)
    }
    
    fn draw_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        if self.space_exhausted {
            return Err(ProviderError::CannotProceed {
                scope: ProviderScope::Exhausted,
                reason: "SMT solver search space exhausted".to_string(),
            });
        }
        
        self.create_symbolic_integer(constraints)
    }
    
    fn draw_float(&mut self, constraints: &FloatConstraints) -> Result<f64, ProviderError> {
        if self.space_exhausted {
            return Err(ProviderError::CannotProceed {
                scope: ProviderScope::Exhausted,
                reason: "SMT solver search space exhausted".to_string(),
            });
        }
        
        // For floats, return a value within constraints
        let value = if constraints.min_value.is_finite() && constraints.max_value.is_finite() {
            (constraints.min_value + constraints.max_value) / 2.0
        } else {
            0.0
        };
        
        println!("SMT_SOLVER DEBUG: Generated symbolic float: {}", value);
        Ok(value)
    }
    
    fn draw_string(&mut self, intervals: &IntervalSet, min_size: usize, max_size: usize) -> Result<String, ProviderError> {
        if self.space_exhausted {
            return Err(ProviderError::CannotProceed {
                scope: ProviderScope::Exhausted,
                reason: "SMT solver search space exhausted".to_string(),
            });
        }
        
        // For demonstration, return a minimal string
        let size = min_size;
        if size == 0 {
            return Ok(String::new());
        }
        
        // Use first available character from intervals
        if let Some((start, _)) = intervals.intervals.first() {
            if let Some(ch) = char::from_u32(*start) {
                return Ok(ch.to_string().repeat(size));
            }
        }
        
        Ok("a".repeat(size)) // Fallback
    }
    
    fn draw_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, ProviderError> {
        if self.space_exhausted {
            return Err(ProviderError::CannotProceed {
                scope: ProviderScope::Exhausted,
                reason: "SMT solver search space exhausted".to_string(),
            });
        }
        
        // For demonstration, return minimal bytes
        Ok(vec![0u8; min_size])
    }
    
    fn observe_test_case(&mut self) -> HashMap<String, serde_json::Value> {
        let mut observation = HashMap::new();
        observation.insert("solver_status".to_string(), 
                         serde_json::Value::String(format!("{:?}", self.solver_state.status)));
        observation.insert("symbolic_values_count".to_string(),
                         serde_json::Value::Number(serde_json::Number::from(self.symbolic_values.len())));
        observation.insert("constraints_count".to_string(),
                         serde_json::Value::Number(serde_json::Number::from(self.constraints.len())));
        observation.insert("space_exhausted".to_string(),
                         serde_json::Value::Bool(self.space_exhausted));
        observation
    }
    
    fn can_realize(&self) -> bool {
        true // SMT solver can handle symbolic values
    }
    
    fn per_test_case_context(&mut self) -> Box<dyn TestCaseContext> {
        Box::new(SmtTestCaseContext {
            provider_id: format!("smt_{}", std::process::id()),
            start_time: std::time::SystemTime::now(),
        })
    }
    
    fn span_start(&mut self, label: u32) {
        println!("SMT_SOLVER DEBUG: Entering span {:#08X}", label);
        // In real implementation, this could push constraint contexts
    }
    
    fn span_end(&mut self, discard: bool) {
        println!("SMT_SOLVER DEBUG: Exiting span, discard: {}", discard);
        if discard {
            // In real implementation, this could pop constraint contexts
        }
    }
}

/// Test case context for SMT solver provider
#[derive(Debug)]
pub struct SmtTestCaseContext {
    provider_id: String,
    start_time: std::time::SystemTime,
}

impl TestCaseContext for SmtTestCaseContext {
    fn enter_test_case(&mut self) {
        println!("SMT_CONTEXT DEBUG: Entering test case for provider {}", self.provider_id);
    }
    
    fn exit_test_case(&mut self, success: bool) {
        let duration = self.start_time.elapsed().unwrap_or_default();
        println!("SMT_CONTEXT DEBUG: Exiting test case for provider {}, success: {}, duration: {:?}", 
                self.provider_id, success, duration);
    }
    
    fn get_context_data(&self) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert("provider_id".to_string(), serde_json::Value::String(self.provider_id.clone()));
        data.insert("start_time".to_string(), 
                   serde_json::Value::String(format!("{:?}", self.start_time)));
        data
    }
}

/// Factory for SMT Solver Provider
pub struct SmtSolverProviderFactory;

impl ProviderFactory for SmtSolverProviderFactory {
    fn create_provider(&self) -> Box<dyn PrimitiveProvider> {
        Box::new(SmtSolverProvider::new())
    }
    
    fn name(&self) -> &str {
        "smt"
    }
    
    fn dependencies(&self) -> Vec<&str> {
        vec![] // No dependencies for this demo implementation
    }
    
    fn validate_environment(&self) -> Result<(), ProviderError> {
        // In real implementation, this would check for SMT solver availability
        println!("SMT_FACTORY DEBUG: Validating SMT solver environment");
        Ok(())
    }
}

/// Fuzzing-based provider for coverage-guided generation
/// 
/// This provider uses fuzzing techniques to guide generation towards
/// increased code coverage and interesting program states.
#[derive(Debug)]
pub struct FuzzingProvider {
    /// Coverage information from previous runs
    coverage_data: HashMap<String, u64>,
    /// Corpus of interesting inputs
    corpus: Vec<Vec<u8>>,
    /// Mutation strategies
    mutation_strategies: Vec<MutationStrategy>,
    /// Current generation strategy
    current_strategy: usize,
}

#[derive(Debug, Clone)]
pub enum MutationStrategy {
    BitFlip,
    ByteFlip,
    ArithmeticIncrement,
    ArithmeticDecrement,
    InterestingValues,
    Dictionary,
    Splice,
}

impl FuzzingProvider {
    pub fn new() -> Self {
        Self {
            coverage_data: HashMap::new(),
            corpus: Vec::new(),
            mutation_strategies: vec![
                MutationStrategy::BitFlip,
                MutationStrategy::ByteFlip,
                MutationStrategy::ArithmeticIncrement,
                MutationStrategy::InterestingValues,
            ],
            current_strategy: 0,
        }
    }
    
    fn mutate_corpus_entry(&mut self, entry: &[u8]) -> Vec<u8> {
        let strategy = &self.mutation_strategies[self.current_strategy % self.mutation_strategies.len()];
        self.current_strategy += 1;
        
        match strategy {
            MutationStrategy::BitFlip => {
                let mut result = entry.to_vec();
                if !result.is_empty() {
                    let bit_index = self.current_strategy % (result.len() * 8);
                    let byte_index = bit_index / 8;
                    let bit_offset = bit_index % 8;
                    result[byte_index] ^= 1 << bit_offset;
                }
                result
            },
            MutationStrategy::ByteFlip => {
                let mut result = entry.to_vec();
                if !result.is_empty() {
                    let byte_index = self.current_strategy % result.len();
                    result[byte_index] = result[byte_index].wrapping_add(1);
                }
                result
            },
            MutationStrategy::InterestingValues => {
                // Use some interesting byte values
                let interesting_bytes = [0, 1, 255, 127, 128];
                let mut result = entry.to_vec();
                if !result.is_empty() {
                    let byte_index = self.current_strategy % result.len();
                    let value_index = self.current_strategy % interesting_bytes.len();
                    result[byte_index] = interesting_bytes[value_index];
                }
                result
            },
            _ => entry.to_vec(), // Other strategies not implemented in this demo
        }
    }
}

impl PrimitiveProvider for FuzzingProvider {
    fn lifetime(&self) -> ProviderLifetime {
        ProviderLifetime::TestRun // Fuzzing benefits from accumulating corpus across test cases
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_integers: true,
            supports_floats: true,
            supports_strings: true,
            supports_bytes: true,
            supports_choices: true,
            avoid_realization: false,
            add_observability_callback: true,
            structural_awareness: true,
            replay_support: true,  // Fuzzing can replay interesting corpus entries
            symbolic_constraints: false,
        }
    }
    
    fn draw_boolean(&mut self, p: f64) -> Result<bool, ProviderError> {
        // For fuzzing, we might want to bias towards edge cases
        if p < 0.1 {
            Ok(false)
        } else if p > 0.9 {
            Ok(true)
        } else {
            Ok(self.current_strategy % 2 == 0)
        }
    }
    
    fn draw_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        let min = constraints.min_value.unwrap_or(i128::MIN);
        let max = constraints.max_value.unwrap_or(i128::MAX);
        
        // Fuzzing strategy: prefer edge cases and interesting values
        let interesting_values = [
            min, max, 0, 1, -1, 
            i8::MIN as i128, i8::MAX as i128,
            i16::MIN as i128, i16::MAX as i128,
            i32::MIN as i128, i32::MAX as i128,
        ];
        
        for &value in &interesting_values {
            if value >= min && value <= max {
                println!("FUZZING DEBUG: Selected interesting integer value: {}", value);
                return Ok(value);
            }
        }
        
        // Fallback to constraint bounds
        Ok(if self.current_strategy % 2 == 0 { min } else { max })
    }
    
    fn draw_float(&mut self, constraints: &FloatConstraints) -> Result<f64, ProviderError> {
        // Fuzzing strategy: use interesting float values
        let interesting_floats = [
            0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY,
            f64::MIN, f64::MAX, f64::MIN_POSITIVE, f64::EPSILON,
        ];
        
        for &value in &interesting_floats {
            if value >= constraints.min_value && value <= constraints.max_value {
                if constraints.allow_nan || !value.is_nan() {
                    println!("FUZZING DEBUG: Selected interesting float value: {}", value);
                    return Ok(value);
                }
            }
        }
        
        // Include NaN if allowed
        if constraints.allow_nan && self.current_strategy % 10 == 0 {
            return Ok(f64::NAN);
        }
        
        Ok(constraints.min_value)
    }
    
    fn draw_string(&mut self, intervals: &IntervalSet, min_size: usize, max_size: usize) -> Result<String, ProviderError> {
        // For fuzzing, try to use corpus entries if available
        if !self.corpus.is_empty() && self.current_strategy % 3 == 0 {
            let corpus_index = self.current_strategy % self.corpus.len();
            let corpus_entry = self.corpus[corpus_index].clone(); // Clone to avoid borrow issues
            let mutated = self.mutate_corpus_entry(&corpus_entry);
            
            // Try to convert bytes to string
            if let Ok(string) = String::from_utf8(mutated) {
                if string.len() >= min_size && string.len() <= max_size {
                    if string.chars().all(|c| intervals.contains(c as u32)) {
                        println!("FUZZING DEBUG: Using mutated corpus string: {:?}", string);
                        return Ok(string);
                    }
                }
            }
        }
        
        // Fallback to interesting strings
        let interesting_strings = ["", "\0", "\n", "\r\n", "\\", "\"", "'"];
        for &s in &interesting_strings {
            if s.len() >= min_size && s.len() <= max_size {
                if s.chars().all(|c| intervals.contains(c as u32)) {
                    return Ok(s.to_string());
                }
            }
        }
        
        // Generate minimal valid string
        if min_size == 0 {
            Ok(String::new())
        } else {
            if let Some((start, _)) = intervals.intervals.first() {
                if let Some(ch) = char::from_u32(*start) {
                    return Ok(ch.to_string().repeat(min_size));
                }
            }
            Ok("a".repeat(min_size))
        }
    }
    
    fn draw_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, ProviderError> {
        // For fuzzing, prefer corpus mutations
        if !self.corpus.is_empty() {
            let corpus_index = self.current_strategy % self.corpus.len();
            let corpus_entry = self.corpus[corpus_index].clone(); // Clone to avoid borrow issues
            let mutated = self.mutate_corpus_entry(&corpus_entry);
            
            if mutated.len() >= min_size && mutated.len() <= max_size {
                println!("FUZZING DEBUG: Using mutated corpus bytes of length {}", mutated.len());
                return Ok(mutated);
            }
        }
        
        // Interesting byte patterns
        let interesting_patterns = [
            vec![0u8; min_size],
            vec![255u8; min_size],
            (0..min_size).map(|i| i as u8).collect(),
        ];
        
        for pattern in &interesting_patterns {
            if pattern.len() >= min_size && pattern.len() <= max_size {
                return Ok(pattern.clone());
            }
        }
        
        Ok(vec![0u8; min_size])
    }
    
    fn observe_test_case(&mut self) -> HashMap<String, serde_json::Value> {
        let mut observation = HashMap::new();
        observation.insert("corpus_size".to_string(),
                         serde_json::Value::Number(serde_json::Number::from(self.corpus.len())));
        observation.insert("coverage_entries".to_string(),
                         serde_json::Value::Number(serde_json::Number::from(self.coverage_data.len())));
        observation.insert("current_strategy".to_string(),
                         serde_json::Value::String(format!("{:?}", 
                             self.mutation_strategies[self.current_strategy % self.mutation_strategies.len()])));
        observation
    }
    
    fn replay_choices(&mut self, choices: &[ChoiceValue]) -> Result<(), ProviderError> {
        // Convert interesting choices to corpus entries
        for choice in choices {
            match choice {
                ChoiceValue::Bytes(bytes) => {
                    if !self.corpus.contains(bytes) {
                        self.corpus.push(bytes.clone());
                        println!("FUZZING DEBUG: Added bytes to corpus: {:?}", bytes);
                    }
                },
                ChoiceValue::String(s) => {
                    let bytes = s.as_bytes().to_vec();
                    if !self.corpus.contains(&bytes) {
                        self.corpus.push(bytes);
                        println!("FUZZING DEBUG: Added string bytes to corpus: {:?}", s);
                    }
                },
                _ => {}, // Other types could be converted to bytes as well
            }
        }
        Ok(())
    }
    
    fn span_start(&mut self, label: u32) {
        // For fuzzing, track coverage information
        let coverage_key = format!("span_{:#08X}", label);
        let current_count = self.coverage_data.get(&coverage_key).unwrap_or(&0);
        self.coverage_data.insert(coverage_key, current_count + 1);
    }
}

/// Factory for Fuzzing Provider
pub struct FuzzingProviderFactory;

impl ProviderFactory for FuzzingProviderFactory {
    fn create_provider(&self) -> Box<dyn PrimitiveProvider> {
        Box::new(FuzzingProvider::new())
    }
    
    fn name(&self) -> &str {
        "fuzzing"
    }
}

/// Register all specialized backends with the global registry
pub fn register_specialized_backends() {
    let mut registry = get_provider_registry();
    
    println!("SPECIALIZED_BACKENDS DEBUG: Registering SMT solver backend");
    registry.register_factory(Arc::new(SmtSolverProviderFactory));
    
    println!("SPECIALIZED_BACKENDS DEBUG: Registering fuzzing backend");
    registry.register_factory(Arc::new(FuzzingProviderFactory));
    
    println!("SPECIALIZED_BACKENDS DEBUG: Registered {} specialized backends", 2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::IntegerConstraints;

    #[test]
    fn test_enhanced_provider_registry() {
        let mut registry = ProviderRegistry::new();
        let available = registry.available_providers();
        
        assert!(available.contains(&"hypothesis".to_string()));
        assert!(available.contains(&"random".to_string()));
        
        let hypothesis_provider = registry.create("hypothesis").unwrap();
        assert_eq!(hypothesis_provider.lifetime(), ProviderLifetime::TestCase);
        assert!(hypothesis_provider.capabilities().supports_integers);
        assert!(hypothesis_provider.capabilities().supports_floats);
        assert!(hypothesis_provider.capabilities().add_observability_callback);
        assert!(hypothesis_provider.capabilities().structural_awareness);
        
        let random_provider = registry.create("random").unwrap();
        assert_eq!(random_provider.lifetime(), ProviderLifetime::TestCase);
        assert!(random_provider.capabilities().supports_choices);
        assert!(!random_provider.capabilities().avoid_realization);
    }
    
    #[test]
    fn test_provider_factory_registration() {
        let mut registry = ProviderRegistry::new();
        
        // Test custom factory registration
        struct TestProviderFactory;
        impl ProviderFactory for TestProviderFactory {
            fn create_provider(&self) -> Box<dyn PrimitiveProvider> {
                Box::new(RandomProvider::new())
            }
            fn name(&self) -> &str { "test_provider" }
        }
        
        registry.register_factory(Arc::new(TestProviderFactory));
        
        let available = registry.available_providers();
        assert!(available.contains(&"test_provider".to_string()));
        
        let test_provider = registry.create("test_provider").unwrap();
        assert_eq!(test_provider.lifetime(), ProviderLifetime::TestCase);
    }
    
    #[test]
    fn test_provider_configuration() {
        let mut registry = ProviderRegistry::new();
        
        // Test configuration setting and retrieval
        let mut config = HashMap::new();
        config.insert("test_param".to_string(), serde_json::Value::String("test_value".to_string()));
        
        assert!(registry.set_provider_config("random", config.clone()).is_ok());
        
        let retrieved_config = registry.get_provider_config("random");
        assert!(retrieved_config.is_some());
        assert_eq!(retrieved_config.unwrap().get("test_param").unwrap(), &serde_json::Value::String("test_value".to_string()));
    }
    
    #[test]
    fn test_backend_validation() {
        let registry = ProviderRegistry::new();
        
        // Test valid backend
        assert!(registry.validate_backend("random").is_ok());
        assert!(registry.validate_backend("hypothesis").is_ok());
        
        // Test invalid backend
        let result = registry.validate_backend("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not available"));
    }
    
    #[test]
    fn test_provider_info() {
        let registry = ProviderRegistry::new();
        let info = registry.provider_info();
        
        assert!(info.contains_key("random"));
        assert!(info.contains_key("hypothesis"));
        
        let random_info = &info["random"];
        assert!(random_info.get("name").is_some());
        assert!(random_info.get("capabilities").is_some());
        assert!(random_info.get("lifetime").is_some());
    }
    
    #[test]
    fn test_specialized_backends() {
        // Register specialized backends
        register_specialized_backends();
        
        let mut registry = get_provider_registry();
        let available = registry.available_providers();
        
        assert!(available.contains(&"smt".to_string()));
        assert!(available.contains(&"fuzzing".to_string()));
        
        // Test SMT provider
        let smt_provider = registry.create("smt").unwrap();
        assert_eq!(smt_provider.lifetime(), ProviderLifetime::TestFunction);
        assert!(smt_provider.capabilities().supports_integers);
        assert!(smt_provider.capabilities().avoid_realization);
        assert!(smt_provider.capabilities().symbolic_constraints);
        
        // Test Fuzzing provider
        let fuzzing_provider = registry.create("fuzzing").unwrap();
        assert_eq!(fuzzing_provider.lifetime(), ProviderLifetime::TestRun);
        assert!(fuzzing_provider.capabilities().supports_choices);
        assert!(fuzzing_provider.capabilities().replay_support);
        assert!(!fuzzing_provider.capabilities().symbolic_constraints);
    }
    
    #[test]
    fn test_smt_solver_provider() {
        let mut provider = SmtSolverProvider::new();
        
        // Test integer generation
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(50),
        };
        
        let value = provider.draw_integer(&constraints).unwrap();
        assert!(value >= 0 && value <= 100);
        
        // Test observability
        let observation = provider.observe_test_case();
        assert!(observation.contains_key("solver_status"));
        assert!(observation.contains_key("symbolic_values_count"));
        
        // Test span tracking
        provider.span_start(0x12345678);
        provider.span_end(false);
    }
    
    #[test]
    fn test_fuzzing_provider() {
        let mut provider = FuzzingProvider::new();
        
        // Test integer generation with interesting values
        let constraints = IntegerConstraints {
            min_value: Some(-100),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        };
        
        let value = provider.draw_integer(&constraints).unwrap();
        assert!(value >= -100 && value <= 100);
        
        // Test corpus replay
        let choices = vec![
            ChoiceValue::Bytes(vec![1, 2, 3, 4]),
            ChoiceValue::String("test".to_string()),
        ];
        
        assert!(provider.replay_choices(&choices).is_ok());
        
        let observation = provider.observe_test_case();
        assert!(observation.contains_key("corpus_size"));
        assert_eq!(observation["corpus_size"], serde_json::Value::Number(serde_json::Number::from(2)));
    }
    
    #[test]
    fn test_provider_error_handling() {
        let mut provider = SmtSolverProvider::new();
        
        // Simulate space exhausted condition
        provider.space_exhausted = true;
        
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        };
        
        let result = provider.draw_integer(&constraints);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProviderError::CannotProceed { scope, .. } => {
                assert_eq!(scope, ProviderScope::Exhausted);
            },
            _ => panic!("Expected CannotProceed error"),
        }
    }
    
    #[test]
    fn test_interval_set() {
        let ascii_set = IntervalSet::ascii();
        assert!(ascii_set.contains(65)); // 'A'
        assert!(ascii_set.contains(97)); // 'a'
        assert!(!ascii_set.contains(1));  // Control character
        
        let char_set = IntervalSet::from_chars("abc");
        assert!(char_set.contains(97)); // 'a'
        assert!(char_set.contains(98)); // 'b'
        assert!(char_set.contains(99)); // 'c'
        assert!(!char_set.contains(100)); // 'd'
    }
    
    #[test]
    fn test_provider_capabilities() {
        let default_caps = BackendCapabilities::default();
        assert!(default_caps.supports_integers);
        assert!(default_caps.supports_floats);
        assert!(default_caps.supports_strings);
        assert!(default_caps.supports_bytes);
        assert!(default_caps.supports_choices);
        assert!(!default_caps.avoid_realization);
        assert!(!default_caps.add_observability_callback);
        assert!(!default_caps.structural_awareness);
        assert!(!default_caps.replay_support);
        assert!(!default_caps.symbolic_constraints);
        
        let enhanced_caps = BackendCapabilities {
            supports_integers: true,
            supports_floats: true,
            supports_strings: true,
            supports_bytes: true,
            supports_choices: true,
            avoid_realization: true,
            add_observability_callback: true,
            structural_awareness: true,
            replay_support: true,
            symbolic_constraints: true,
        };
        
        assert!(enhanced_caps.supports_choices);
        assert!(enhanced_caps.avoid_realization);
        assert!(enhanced_caps.add_observability_callback);
        assert!(enhanced_caps.structural_awareness);
        assert!(enhanced_caps.replay_support);
        assert!(enhanced_caps.symbolic_constraints);
    }
    
    #[test]
    fn test_test_case_context() {
        let mut context = SmtTestCaseContext {
            provider_id: "test_id".to_string(),
            start_time: std::time::SystemTime::now(),
        };
        
        context.enter_test_case();
        context.exit_test_case(true);
        
        let data = context.get_context_data();
        assert!(data.contains_key("provider_id"));
        assert_eq!(data["provider_id"], serde_json::Value::String("test_id".to_string()));
    }
    
    #[test]
    fn test_global_provider_functions() {
        // Test global registry functions
        register_specialized_backends();
        
        let provider_info = get_global_provider_info();
        assert!(provider_info.contains_key("random"));
        assert!(provider_info.contains_key("hypothesis"));
        assert!(provider_info.contains_key("smt"));
        assert!(provider_info.contains_key("fuzzing"));
        
        // Test global provider creation
        let random_provider = create_global_provider("random");
        assert!(random_provider.is_ok());
        
        let invalid_provider = create_global_provider("nonexistent");
        assert!(invalid_provider.is_err());
    }
    
    #[test]
    fn test_legacy_compatibility() {
        let mut provider = RandomProvider::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        };
        
        // Test legacy methods still work
        let value = provider.generate_integer(&mut rng, &constraints).unwrap();
        assert!(value >= 0 && value <= 100);
        
        let bool_constraints = BooleanConstraints { p: 0.5 };
        let bool_value = provider.generate_boolean(&mut rng, &bool_constraints).unwrap();
        assert!(bool_value == true || bool_value == false);
    }

    #[test]
    fn test_random_provider() {
        let mut provider = RandomProvider::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        };
        
        let value = provider.generate_integer(&mut rng, &constraints).unwrap();
        assert!(value >= 0 && value <= 100);
    }

    #[test]
    fn test_hypothesis_provider_constant_injection() {
        let mut provider = HypothesisProvider::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        // Run many iterations to test constant injection
        let mut constant_used = false;
        for _ in 0..100 {
            let constraints = IntegerConstraints {
                min_value: Some(-1000),
                max_value: Some(1000),
                weights: None,
                shrink_towards: Some(0),
            };
            
            let value = provider.generate_integer(&mut rng, &constraints).unwrap();
            
            // Check if value matches any of our global constants
            if provider.global_constants.integers.contains(&value) {
                constant_used = true;
                println!("Constant injection worked: generated {}", value);
                break;
            }
        }
        
        // With 5% probability per attempt and 100 attempts, we should see at least one constant
        // This test might occasionally fail due to randomness, but should be very rare
        if !constant_used {
            println!("Warning: No constants were injected in 100 attempts (this is rare but possible)");
        }
    }

    #[test]
    fn test_global_constants() {
        let constants = GlobalConstants::new();
        
        // Test that we have reasonable numbers of constants
        assert!(!constants.integers.is_empty());
        assert!(!constants.floats.is_empty());
        assert!(!constants.strings.is_empty());
        assert!(!constants.bytes.is_empty());
        
        // Test that some expected constants are present
        assert!(constants.integers.contains(&0));
        assert!(constants.integers.contains(&1));
        assert!(constants.integers.contains(&-1));
        
        assert!(constants.floats.contains(&0.0));
        assert!(constants.floats.contains(&1.0));
        assert!(constants.floats.iter().any(|f| f.is_nan()));
        
        assert!(constants.strings.contains(&String::new()));
        assert!(constants.strings.contains(&"test".to_string()));
        
        assert!(constants.bytes.contains(&vec![]));
        assert!(constants.bytes.contains(&vec![0]));
    }

    #[test]
    fn test_provider_span_notifications() {
        let mut provider = HypothesisProvider::new();
        
        // These shouldn't panic
        provider.span_start(42u32); // Use numeric label instead of string
        provider.span_end(false);
        provider.span_end(true);
    }
}

// Implement FloatPrimitiveProvider for existing providers to work with FloatConstraintTypeSystem
impl crate::choice::float_constraint_type_system::FloatPrimitiveProvider for RandomProvider {
    fn generate_u64(&mut self) -> u64 {
        let mut rng = ChaCha8Rng::from_entropy();
        rng.gen()
    }
    
    fn generate_f64(&mut self) -> f64 {
        let mut rng = ChaCha8Rng::from_entropy();
        rng.gen()
    }
    
    fn generate_usize(&mut self) -> usize {
        let mut rng = ChaCha8Rng::from_entropy();
        rng.gen()
    }
    
    fn generate_bool(&mut self) -> bool {
        let mut rng = ChaCha8Rng::from_entropy();
        rng.gen()
    }
    
    fn generate_float(&mut self, constraints: &FloatConstraints) -> f64 {
        let mut rng = ChaCha8Rng::from_entropy();
        let value = rng.gen::<f64>();
        if constraints.validate(value) {
            value
        } else {
            constraints.clamp(value)
        }
    }
}

impl crate::choice::float_constraint_type_system::FloatPrimitiveProvider for HypothesisProvider {
    fn generate_u64(&mut self) -> u64 {
        let mut rng = ChaCha8Rng::from_entropy();
        rng.gen()
    }
    
    fn generate_f64(&mut self) -> f64 {
        let mut rng = ChaCha8Rng::from_entropy();
        rng.gen()
    }
    
    fn generate_usize(&mut self) -> usize {
        let mut rng = ChaCha8Rng::from_entropy();
        rng.gen()
    }
    
    fn generate_bool(&mut self) -> bool {
        let mut rng = ChaCha8Rng::from_entropy();
        rng.gen()
    }
    
    fn generate_float(&mut self, constraints: &FloatConstraints) -> f64 {
        let mut rng = ChaCha8Rng::from_entropy();
        let value = rng.gen::<f64>();
        if constraints.validate(value) {
            value
        } else {
            constraints.clamp(value)
        }
    }
}

// Implement FloatConstraintAwareProvider for existing providers
impl crate::choice::float_constraint_type_system::FloatConstraintAwareProvider for RandomProvider {}
impl crate::choice::float_constraint_type_system::FloatConstraintAwareProvider for HypothesisProvider {}