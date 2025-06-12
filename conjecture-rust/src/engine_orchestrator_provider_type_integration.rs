//! Provider Type System Integration for EngineOrchestrator
//!
//! This module implements the complete Provider Type System Integration capability,
//! fixing critical type mismatches between generic `P: PrimitiveProvider` constraints
//! and `Box<dyn PrimitiveProvider>` returns in provider context switching.
//!
//! Key architectural improvements:
//! - Unified provider interface with consistent method signatures
//! - Dynamic provider instantiation through trait objects
//! - Type-safe provider context switching
//! - Comprehensive error handling for provider lifecycle
//! - Debug logging with uppercase hex notation

use std::collections::HashMap;
use std::fmt::Debug;

use crate::choice::{ChoiceType, ChoiceValue, Constraints};
use crate::data::DrawError;
use crate::providers::{ProviderLifetime, GlobalConstants};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Enhanced PrimitiveProvider trait with unified interface
/// 
/// This trait provides a consistent interface for all provider implementations,
/// resolving the type mismatch issues between static and dynamic dispatch.
pub trait EnhancedPrimitiveProvider: Debug + Send + Sync {
    /// Provider name for identification
    fn name(&self) -> &str;
    
    /// Provider lifecycle management
    fn lifetime(&self) -> ProviderLifetime;
    
    /// Core choice generation method with type safety
    fn draw_choice(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Result<ChoiceValue, DrawError>;
    
    /// Weighted choice selection for complex generation
    fn weighted_choice(&mut self, weights: &[f64]) -> Result<usize, DrawError>;
    
    /// Integer generation with constraints
    fn draw_integer(&mut self, min: i64, max: i64) -> Result<i64, DrawError>;
    
    /// Float generation with constraints  
    fn draw_float(&mut self, min: f64, max: f64) -> Result<f64, DrawError>;
    
    /// Bytes generation with length
    fn draw_bytes(&mut self, length: usize) -> Result<Vec<u8>, DrawError>;
    
    /// String generation with alphabet and size constraints
    fn draw_string(&mut self, alphabet: &str, min_size: usize, max_size: usize) -> Result<String, DrawError>;
    
    /// Boolean generation with probability
    fn draw_boolean(&mut self, probability: f64) -> Result<bool, DrawError>;
    
    /// Span lifecycle notifications for structure awareness
    fn span_start(&mut self, label: &str) {
        eprintln!("PROVIDER DEBUG: Span started: {}", label);
    }
    
    fn span_end(&mut self, discard: bool) {
        eprintln!("PROVIDER DEBUG: Span ended, discard: {}", discard);
    }
    
    /// Provider-specific observation callback
    fn observe(&mut self, event: &str, data: &str) {
        eprintln!("PROVIDER DEBUG: [{}] {} - {}", self.name(), event, data);
    }
}

// ProviderLifetime is imported from crate::providers

/// Provider context for dynamic dispatch and lifecycle management
#[derive(Debug, Clone)]
pub struct ProviderTypeContext {
    /// Current active provider name
    pub active_provider: String,
    /// Provider switching state
    pub switch_to_hypothesis: bool,
    /// Failed realize count for threshold switching
    pub failed_realize_count: usize,
    /// Backend that verified the test case
    pub verified_by: Option<String>,
    /// Provider observation callbacks
    pub observation_callbacks: Vec<String>,
    /// Provider creation timestamp for lifecycle management
    pub created_at: std::time::Instant,
    /// Last switch timestamp
    pub last_switch_at: Option<std::time::Instant>,
}

impl Default for ProviderTypeContext {
    fn default() -> Self {
        Self {
            active_provider: "hypothesis".to_string(),
            switch_to_hypothesis: false,
            failed_realize_count: 0,
            verified_by: None,
            observation_callbacks: Vec::new(),
            created_at: std::time::Instant::now(),
            last_switch_at: None,
        }
    }
}

/// Enhanced provider registry with type-safe dynamic dispatch
pub struct ProviderTypeRegistry {
    providers: HashMap<String, Box<dyn Fn() -> Box<dyn EnhancedPrimitiveProvider> + Send + Sync>>,
}

impl ProviderTypeRegistry {
    /// Create a new type-safe provider registry
    pub fn new() -> Self {
        let mut registry = Self {
            providers: HashMap::new(),
        };
        
        // Register default providers with enhanced interface
        registry.register("hypothesis", || Box::new(EnhancedHypothesisProvider::new()));
        registry.register("random", || Box::new(EnhancedRandomProvider::new()));
        
        registry
    }
    
    /// Register a provider factory with type safety
    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn EnhancedPrimitiveProvider> + Send + Sync + 'static,
    {
        eprintln!("PROVIDER TYPE SYSTEM: Registering provider '{}'", name);
        self.providers.insert(name.to_string(), Box::new(factory));
    }
    
    /// Create a provider instance with dynamic dispatch
    pub fn create(&self, name: &str) -> Option<Box<dyn EnhancedPrimitiveProvider>> {
        eprintln!("PROVIDER TYPE SYSTEM: Creating provider instance '{}'", name);
        self.providers.get(name).map(|factory| {
            let provider = factory();
            eprintln!("PROVIDER TYPE SYSTEM: Successfully created provider '{}'", provider.name());
            provider
        })
    }
    
    /// Get available provider names
    pub fn available_providers(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }
    
    /// Validate provider availability
    pub fn validate_provider(&self, name: &str) -> bool {
        self.providers.contains_key(name)
    }
}

/// Enhanced Hypothesis provider with unified interface
#[derive(Debug)]
pub struct EnhancedHypothesisProvider {
    name: String,
    rng: ChaCha8Rng,
    constants: GlobalConstants,
    local_constants: HashMap<String, Vec<ChoiceValue>>,
}

impl EnhancedHypothesisProvider {
    pub fn new() -> Self {
        Self {
            name: "hypothesis".to_string(),
            rng: ChaCha8Rng::from_entropy(),
            constants: GlobalConstants::new(),
            local_constants: HashMap::new(),
        }
    }
    
    /// Maybe draw from constant pool (5% probability)
    fn maybe_draw_constant(&mut self, choice_type: &str, constraints: &Constraints) -> Option<ChoiceValue> {
        if self.rng.gen::<f64>() < 0.05 {
            eprintln!("PROVIDER TYPE SYSTEM: Drawing constant for {}", choice_type);
            return self.draw_from_constants(choice_type, constraints);
        }
        None
    }
    
    /// Draw from appropriate constant pool
    fn draw_from_constants(&mut self, choice_type: &str, constraints: &Constraints) -> Option<ChoiceValue> {
        match choice_type {
            "integer" => {
                if let Constraints::Integer(int_constraints) = constraints {
                    let min = int_constraints.min_value.unwrap_or(i128::MIN);
                    let max = int_constraints.max_value.unwrap_or(i128::MAX);
                    
                    let valid_constants: Vec<i128> = self.constants.integers.iter()
                        .filter(|&&c| c >= min && c <= max)
                        .copied()
                        .collect();
                    
                    if !valid_constants.is_empty() {
                        let index = self.rng.gen_range(0..valid_constants.len());
                        return Some(ChoiceValue::Integer(valid_constants[index]));
                    }
                }
            }
            "float" => {
                if let Constraints::Float(float_constraints) = constraints {
                    let valid_constants: Vec<f64> = self.constants.floats.iter()
                        .filter(|&&c| c >= float_constraints.min_value && c <= float_constraints.max_value)
                        .filter(|&&c| !c.is_nan() || float_constraints.allow_nan)
                        .copied()
                        .collect();
                    
                    if !valid_constants.is_empty() {
                        let index = self.rng.gen_range(0..valid_constants.len());
                        return Some(ChoiceValue::Float(valid_constants[index]));
                    }
                }
            }
            "boolean" => {
                return Some(ChoiceValue::Boolean(self.rng.gen::<bool>()));
            }
            _ => {}
        }
        None
    }
}

impl EnhancedPrimitiveProvider for EnhancedHypothesisProvider {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn lifetime(&self) -> ProviderLifetime {
        ProviderLifetime::TestCase
    }
    
    fn draw_choice(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Result<ChoiceValue, DrawError> {
        let type_str = match choice_type {
            ChoiceType::Integer => "integer",
            ChoiceType::Float => "float", 
            ChoiceType::Boolean => "boolean",
            ChoiceType::String => "string",
            ChoiceType::Bytes => "bytes",
        };
        
        // Try constant injection first
        if let Some(constant) = self.maybe_draw_constant(type_str, constraints) {
            return Ok(constant);
        }
        
        // Fall back to random generation
        match (choice_type, constraints) {
            (ChoiceType::Integer, Constraints::Integer(int_constraints)) => {
                let min = int_constraints.min_value.unwrap_or(i128::MIN);
                let max = int_constraints.max_value.unwrap_or(i128::MAX);
                
                if min > max {
                    return Err(DrawError::InvalidRange);
                }
                
                let value = if min == max {
                    min
                } else {
                    self.rng.gen_range(min..=max)
                };
                
                Ok(ChoiceValue::Integer(value))
            }
            (ChoiceType::Float, Constraints::Float(float_constraints)) => {
                let value = self.rng.gen_range(float_constraints.min_value..=float_constraints.max_value);
                Ok(ChoiceValue::Float(value))
            }
            (ChoiceType::Boolean, Constraints::Boolean(bool_constraints)) => {
                if bool_constraints.p < 0.0 || bool_constraints.p > 1.0 {
                    return Err(DrawError::InvalidProbability);
                }
                let value = self.rng.gen::<f64>() < bool_constraints.p;
                Ok(ChoiceValue::Boolean(value))
            }
            _ => Err(DrawError::InvalidChoice),
        }
    }
    
    fn weighted_choice(&mut self, weights: &[f64]) -> Result<usize, DrawError> {
        if weights.is_empty() {
            return Err(DrawError::EmptyWeights);
        }
        
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(DrawError::InvalidWeights);
        }
        
        let mut threshold = self.rng.gen::<f64>() * total;
        for (i, &weight) in weights.iter().enumerate() {
            threshold -= weight;
            if threshold <= 0.0 {
                return Ok(i);
            }
        }
        
        Ok(weights.len() - 1) // Fallback to last index
    }
    
    fn draw_integer(&mut self, min: i64, max: i64) -> Result<i64, DrawError> {
        if min > max {
            return Err(DrawError::InvalidRange);
        }
        
        if min == max {
            Ok(min)
        } else {
            Ok(self.rng.gen_range(min..=max))
        }
    }
    
    fn draw_float(&mut self, min: f64, max: f64) -> Result<f64, DrawError> {
        if min > max {
            return Err(DrawError::InvalidRange);
        }
        
        Ok(self.rng.gen_range(min..=max))
    }
    
    fn draw_bytes(&mut self, length: usize) -> Result<Vec<u8>, DrawError> {
        let mut bytes = vec![0u8; length];
        for byte in &mut bytes {
            *byte = self.rng.gen();
        }
        Ok(bytes)
    }
    
    fn draw_string(&mut self, alphabet: &str, min_size: usize, max_size: usize) -> Result<String, DrawError> {
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
            self.rng.gen_range(min_size..=max_size)
        };
        
        let mut result = String::new();
        for _ in 0..size {
            let char_index = self.rng.gen_range(0..alphabet_chars.len());
            result.push(alphabet_chars[char_index]);
        }
        
        Ok(result)
    }
    
    fn draw_boolean(&mut self, probability: f64) -> Result<bool, DrawError> {
        if probability < 0.0 || probability > 1.0 {
            return Err(DrawError::InvalidProbability);
        }
        Ok(self.rng.gen::<f64>() < probability)
    }
}

/// Enhanced Random provider with unified interface
#[derive(Debug)]
pub struct EnhancedRandomProvider {
    name: String,
    rng: ChaCha8Rng,
}

impl EnhancedRandomProvider {
    pub fn new() -> Self {
        Self {
            name: "random".to_string(),
            rng: ChaCha8Rng::from_entropy(),
        }
    }
}

impl EnhancedPrimitiveProvider for EnhancedRandomProvider {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn lifetime(&self) -> ProviderLifetime {
        ProviderLifetime::TestCase
    }
    
    fn draw_choice(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Result<ChoiceValue, DrawError> {
        match (choice_type, constraints) {
            (ChoiceType::Integer, Constraints::Integer(int_constraints)) => {
                let min = int_constraints.min_value.unwrap_or(i128::MIN);
                let max = int_constraints.max_value.unwrap_or(i128::MAX);
                
                if min > max {
                    return Err(DrawError::InvalidRange);
                }
                
                let value = if min == max {
                    min
                } else {
                    self.rng.gen_range(min..=max)
                };
                
                Ok(ChoiceValue::Integer(value))
            }
            (ChoiceType::Float, Constraints::Float(float_constraints)) => {
                let value = self.rng.gen_range(float_constraints.min_value..=float_constraints.max_value);
                Ok(ChoiceValue::Float(value))
            }
            (ChoiceType::Boolean, Constraints::Boolean(bool_constraints)) => {
                if bool_constraints.p < 0.0 || bool_constraints.p > 1.0 {
                    return Err(DrawError::InvalidProbability);
                }
                let value = self.rng.gen::<f64>() < bool_constraints.p;
                Ok(ChoiceValue::Boolean(value))
            }
            _ => Err(DrawError::InvalidChoice),
        }
    }
    
    fn weighted_choice(&mut self, weights: &[f64]) -> Result<usize, DrawError> {
        if weights.is_empty() {
            return Err(DrawError::EmptyWeights);
        }
        
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(DrawError::InvalidWeights);
        }
        
        let mut threshold = self.rng.gen::<f64>() * total;
        for (i, &weight) in weights.iter().enumerate() {
            threshold -= weight;
            if threshold <= 0.0 {
                return Ok(i);
            }
        }
        
        Ok(weights.len() - 1)
    }
    
    fn draw_integer(&mut self, min: i64, max: i64) -> Result<i64, DrawError> {
        if min > max {
            return Err(DrawError::InvalidRange);
        }
        
        if min == max {
            Ok(min)
        } else {
            Ok(self.rng.gen_range(min..=max))
        }
    }
    
    fn draw_float(&mut self, min: f64, max: f64) -> Result<f64, DrawError> {
        if min > max {
            return Err(DrawError::InvalidRange);
        }
        
        Ok(self.rng.gen_range(min..=max))
    }
    
    fn draw_bytes(&mut self, length: usize) -> Result<Vec<u8>, DrawError> {
        let mut bytes = vec![0u8; length];
        for byte in &mut bytes {
            *byte = self.rng.gen();
        }
        Ok(bytes)
    }
    
    fn draw_string(&mut self, alphabet: &str, min_size: usize, max_size: usize) -> Result<String, DrawError> {
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
            self.rng.gen_range(min_size..=max_size)
        };
        
        let mut result = String::new();
        for _ in 0..size {
            let char_index = self.rng.gen_range(0..alphabet_chars.len());
            result.push(alphabet_chars[char_index]);
        }
        
        Ok(result)
    }
    
    fn draw_boolean(&mut self, probability: f64) -> Result<bool, DrawError> {
        if probability < 0.0 || probability > 1.0 {
            return Err(DrawError::InvalidProbability);
        }
        Ok(self.rng.gen::<f64>() < probability)
    }
}

// GlobalConstants is imported from crate::providers

/// Provider type system integration errors
#[derive(Debug, Clone)]
pub enum ProviderTypeError {
    ProviderNotFound { name: String },
    ProviderCreationFailed { name: String, reason: String },
    ProviderSwitchFailed { from: String, to: String, reason: String },
    InvalidProviderType { expected: String, actual: String },
    ProviderContextCorrupted { details: String },
}

impl std::fmt::Display for ProviderTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderTypeError::ProviderNotFound { name } => {
                write!(f, "Provider '{}' not found in registry", name)
            }
            ProviderTypeError::ProviderCreationFailed { name, reason } => {
                write!(f, "Failed to create provider '{}': {}", name, reason)
            }
            ProviderTypeError::ProviderSwitchFailed { from, to, reason } => {
                write!(f, "Failed to switch provider from '{}' to '{}': {}", from, to, reason)
            }
            ProviderTypeError::InvalidProviderType { expected, actual } => {
                write!(f, "Invalid provider type: expected '{}', got '{}'", expected, actual)
            }
            ProviderTypeError::ProviderContextCorrupted { details } => {
                write!(f, "Provider context corrupted: {}", details)
            }
        }
    }
}

impl std::error::Error for ProviderTypeError {}

/// Provider type system manager for orchestrator integration
pub struct ProviderTypeManager {
    registry: ProviderTypeRegistry,
    context: ProviderTypeContext,
    active_provider: Option<Box<dyn EnhancedPrimitiveProvider>>,
}

impl ProviderTypeManager {
    /// Create a new provider type manager
    pub fn new() -> Self {
        Self {
            registry: ProviderTypeRegistry::new(),
            context: ProviderTypeContext::default(),
            active_provider: None,
        }
    }
    
    /// Initialize with specific backend
    pub fn initialize(&mut self, backend: &str) -> Result<(), ProviderTypeError> {
        eprintln!("PROVIDER TYPE SYSTEM: Initializing with backend '{}'", backend);
        
        if !self.registry.validate_provider(backend) {
            return Err(ProviderTypeError::ProviderNotFound {
                name: backend.to_string(),
            });
        }
        
        let provider = self.registry.create(backend)
            .ok_or_else(|| ProviderTypeError::ProviderCreationFailed {
                name: backend.to_string(),
                reason: "Registry creation failed".to_string(),
            })?;
        
        self.context.active_provider = backend.to_string();
        self.active_provider = Some(provider);
        
        eprintln!("PROVIDER TYPE SYSTEM: Successfully initialized with '{}'", backend);
        Ok(())
    }
    
    /// Switch to hypothesis provider
    pub fn switch_to_hypothesis(&mut self) -> Result<(), ProviderTypeError> {
        if self.context.switch_to_hypothesis {
            eprintln!("PROVIDER TYPE SYSTEM: Already using hypothesis provider");
            return Ok(());
        }
        
        let previous = self.context.active_provider.clone();
        eprintln!("PROVIDER TYPE SYSTEM: Switching from '{}' to 'hypothesis'", previous);
        
        let hypothesis_provider = self.registry.create("hypothesis")
            .ok_or_else(|| ProviderTypeError::ProviderSwitchFailed {
                from: previous.clone(),
                to: "hypothesis".to_string(),
                reason: "Failed to create hypothesis provider".to_string(),
            })?;
        
        self.context.active_provider = "hypothesis".to_string();
        self.context.switch_to_hypothesis = true;
        self.context.last_switch_at = Some(std::time::Instant::now());
        self.active_provider = Some(hypothesis_provider);
        
        eprintln!("PROVIDER TYPE SYSTEM: Successfully switched to hypothesis");
        Ok(())
    }
    
    /// Get active provider reference
    pub fn active_provider(&mut self) -> Result<&mut dyn EnhancedPrimitiveProvider, ProviderTypeError> {
        match self.active_provider.as_mut() {
            Some(provider) => Ok(provider.as_mut()),
            None => Err(ProviderTypeError::ProviderContextCorrupted {
                details: "No active provider available".to_string(),
            }),
        }
    }
    
    /// Get provider context
    pub fn context(&self) -> &ProviderTypeContext {
        &self.context
    }
    
    /// Update context for backend cannot proceed
    pub fn handle_backend_cannot_proceed(&mut self, scope: &str) -> Result<(), ProviderTypeError> {
        eprintln!("PROVIDER TYPE SYSTEM: BackendCannotProceed with scope '{}'", scope);
        
        match scope {
            "verified" | "exhausted" => {
                if scope == "verified" {
                    self.context.verified_by = Some(self.context.active_provider.clone());
                }
                self.switch_to_hypothesis()?;
            }
            "discard_test_case" => {
                self.context.failed_realize_count += 1;
                
                // Switch if threshold exceeded (simplified logic)
                if self.context.failed_realize_count > 10 {
                    eprintln!("PROVIDER TYPE SYSTEM: Failed realize threshold exceeded, switching");
                    self.switch_to_hypothesis()?;
                }
            }
            _ => {
                return Err(ProviderTypeError::InvalidProviderType {
                    expected: "verified|exhausted|discard_test_case".to_string(),
                    actual: scope.to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    /// Add observation callback
    pub fn register_observation_callback(&mut self, callback_id: String) {
        eprintln!("PROVIDER TYPE SYSTEM: Registering callback '{}'", callback_id);
        self.context.observation_callbacks.push(callback_id);
    }
    
    /// Log provider observation with hex ID
    pub fn log_observation(&self, event: &str, details: &str) {
        let timestamp = self.context.created_at.elapsed().as_millis();
        let hex_id = format!("{:08X}", timestamp); // Uppercase hex notation
        eprintln!("PROVIDER TYPE SYSTEM: [{}] {} - {}", hex_id, event, details);
    }
    
    /// Cleanup provider resources
    pub fn cleanup(&mut self) {
        eprintln!("PROVIDER TYPE SYSTEM: Cleaning up provider resources");
        
        if let Some(provider) = &self.active_provider {
            eprintln!("PROVIDER TYPE SYSTEM: Cleaning up provider '{}'", provider.name());
        }
        
        self.log_observation("cleanup_started", &format!(
            "active_provider={}, switched={}, failed_realizes={}",
            self.context.active_provider,
            self.context.switch_to_hypothesis,
            self.context.failed_realize_count
        ));
        
        self.context.observation_callbacks.clear();
        self.active_provider = None;
        
        eprintln!("PROVIDER TYPE SYSTEM: Cleanup completed");
    }
}

impl Drop for ProviderTypeManager {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_provider_type_registry() {
        let registry = ProviderTypeRegistry::new();
        let available = registry.available_providers();
        
        assert!(available.contains(&"hypothesis".to_string()));
        assert!(available.contains(&"random".to_string()));
        
        let hypothesis_provider = registry.create("hypothesis").unwrap();
        assert_eq!(hypothesis_provider.name(), "hypothesis");
        
        let random_provider = registry.create("random").unwrap();
        assert_eq!(random_provider.name(), "random");
    }
    
    #[test]
    fn test_enhanced_hypothesis_provider() {
        let mut provider = EnhancedHypothesisProvider::new();
        
        assert_eq!(provider.name(), "hypothesis");
        assert_eq!(provider.lifetime(), ProviderLifetime::TestCase);
        
        let weights = vec![0.5, 0.3, 0.2];
        let choice = provider.weighted_choice(&weights);
        assert!(choice.is_ok());
        assert!(choice.unwrap() < 3);
        
        let integer = provider.draw_integer(-10, 10);
        assert!(integer.is_ok());
        let val = integer.unwrap();
        assert!(val >= -10 && val <= 10);
    }
    
    #[test]
    fn test_provider_type_manager() {
        let mut manager = ProviderTypeManager::new();
        
        // Test initialization
        let init_result = manager.initialize("hypothesis");
        assert!(init_result.is_ok());
        assert_eq!(manager.context().active_provider, "hypothesis");
        
        // Test provider access
        let provider = manager.active_provider();
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().name(), "hypothesis");
        
        // Test switching (should be no-op since already hypothesis)
        let switch_result = manager.switch_to_hypothesis();
        assert!(switch_result.is_ok());
    }
    
    #[test]
    fn test_backend_cannot_proceed_handling() {
        let mut manager = ProviderTypeManager::new();
        manager.initialize("random").unwrap();
        
        // Test verified scope
        let result = manager.handle_backend_cannot_proceed("verified");
        assert!(result.is_ok());
        assert!(manager.context().switch_to_hypothesis);
        assert_eq!(manager.context().verified_by, Some("random".to_string()));
        
        // Test discard scope
        let mut manager2 = ProviderTypeManager::new();
        manager2.initialize("random").unwrap();
        
        for _ in 0..12 {
            let result = manager2.handle_backend_cannot_proceed("discard_test_case");
            assert!(result.is_ok());
        }
        
        assert!(manager2.context().switch_to_hypothesis);
        assert_eq!(manager2.context().failed_realize_count, 12);
    }
    
    #[test]
    fn test_provider_observation_system() {
        let mut manager = ProviderTypeManager::new();
        manager.initialize("hypothesis").unwrap();
        
        // Test callback registration
        manager.register_observation_callback("test_callback".to_string());
        assert_eq!(manager.context().observation_callbacks.len(), 1);
        
        // Test observation logging
        manager.log_observation("test_event", "test_details");
        
        // Test cleanup
        manager.cleanup();
        assert!(manager.context().observation_callbacks.is_empty());
    }
}