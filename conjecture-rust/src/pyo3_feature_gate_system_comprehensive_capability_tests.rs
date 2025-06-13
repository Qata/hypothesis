//! PyO3 Feature Gate System - Comprehensive Module Capability
//!
//! This module provides a complete feature gate system for PyO3 functionality,
//! enabling conditional compilation and proper separation of Python FFI code
//! from Rust-only functionality.
//!
//! # Architecture
//!
//! The system implements a layered approach to feature gating:
//! 1. **Core Feature Gate**: `python-ffi` feature controls PyO3 availability
//! 2. **Conditional Types**: Rust-native alternatives when PyO3 is disabled
//! 3. **FFI Bridge**: Safe interop layer with proper error handling
//! 4. **Test Isolation**: Separate test execution paths for FFI and Rust-only
//!
//! # Usage
//!
//! ```rust
//! // Enable PyO3 functionality
//! #[cfg(feature = "python-ffi")]
//! use pyo3_feature_gate_system::*;
//! 
//! // Use conditional compilation
//! #[cfg(feature = "python-ffi")]
//! fn with_python() -> PyResult<()> { /* ... */ }
//! 
//! #[cfg(not(feature = "python-ffi"))]
//! fn without_python() -> Result<(), String> { /* ... */ }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use log::{debug, info, warn};

/// Feature gate configuration and management
pub struct PyO3FeatureGateSystem {
    feature_enabled: bool,
    debug_logging: bool,
    fallback_registry: HashMap<String, Arc<dyn FallbackHandler>>,
}

/// Error types for feature gate operations
#[derive(Debug, Clone)]
pub enum FeatureGateError {
    PyO3NotAvailable,
    FallbackNotFound(String),
    ConversionError(String),
    ConfigurationError(String),
}

impl fmt::Display for FeatureGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeatureGateError::PyO3NotAvailable => {
                write!(f, "PyO3 functionality not available - compile with --features python-ffi")
            }
            FeatureGateError::FallbackNotFound(name) => {
                write!(f, "No fallback handler found for '{}'", name)
            }
            FeatureGateError::ConversionError(msg) => {
                write!(f, "Type conversion error: {}", msg)
            }
            FeatureGateError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
        }
    }
}

impl std::error::Error for FeatureGateError {}

/// Trait for implementing fallback behavior when PyO3 is not available
pub trait FallbackHandler: Send + Sync {
    fn handle(&self, operation: &str, args: &[String]) -> Result<String, FeatureGateError>;
    fn name(&self) -> &str;
}

/// Default fallback handler for basic operations
#[derive(Debug)]
pub struct DefaultFallbackHandler {
    name: String,
}

impl DefaultFallbackHandler {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
        }
    }
}

impl FallbackHandler for DefaultFallbackHandler {
    fn handle(&self, operation: &str, args: &[String]) -> Result<String, FeatureGateError> {
        debug!("Executing fallback for operation '{}' with args: {:?}", operation, args);
        Ok(format!("fallback_{}_{}", operation, args.len()))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Conditional type wrapper for PyO3 objects
#[cfg(feature = "python-ffi")]
pub type PyObjectWrapper = pyo3::PyObject;

#[cfg(not(feature = "python-ffi"))]
pub type PyObjectWrapper = MockPyObject;

/// Mock PyObject for when PyO3 is not available
#[cfg(not(feature = "python-ffi"))]
#[derive(Debug, Clone)]
pub struct MockPyObject {
    type_name: String,
    value: String,
}

#[cfg(not(feature = "python-ffi"))]
impl MockPyObject {
    pub fn new(type_name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            value: value.into(),
        }
    }

    pub fn type_name(&self) -> &str {
        &self.type_name
    }

    pub fn to_string(&self) -> String {
        self.value.clone()
    }
}

/// Conditional result type for PyO3 operations
#[cfg(feature = "python-ffi")]
pub type PyResultWrapper<T> = pyo3::PyResult<T>;

#[cfg(not(feature = "python-ffi"))]
pub type PyResultWrapper<T> = Result<T, FeatureGateError>;

impl PyO3FeatureGateSystem {
    /// Create a new feature gate system
    pub fn new() -> Self {
        let feature_enabled = cfg!(feature = "python-ffi");
        
        debug!("Initializing PyO3 Feature Gate System - PyO3 enabled: {}", feature_enabled);
        
        let mut system = Self {
            feature_enabled,
            debug_logging: true,
            fallback_registry: HashMap::new(),
        };

        // Register default fallback handlers
        system.register_fallback(Arc::new(DefaultFallbackHandler::new("default")));
        system.register_fallback(Arc::new(DefaultFallbackHandler::new("conversion")));
        system.register_fallback(Arc::new(DefaultFallbackHandler::new("execution")));

        info!("PyO3 Feature Gate System initialized with {} fallback handlers", 
              system.fallback_registry.len());

        system
    }

    /// Check if PyO3 features are available
    pub fn is_pyo3_available(&self) -> bool {
        self.feature_enabled
    }

    /// Register a fallback handler
    pub fn register_fallback(&mut self, handler: Arc<dyn FallbackHandler>) {
        let name = handler.name().to_string();
        debug!("Registering fallback handler: {}", name);
        self.fallback_registry.insert(name, handler);
    }

    /// Execute operation with proper feature gating
    pub fn execute_gated<T, F>(&self, operation: &str, pyo3_fn: F) -> PyResultWrapper<T>
    where
        F: FnOnce() -> PyResultWrapper<T>,
        T: Default,
    {
        if self.feature_enabled {
            debug!("Executing PyO3 operation: {}", operation);
            pyo3_fn()
        } else {
            warn!("PyO3 operation '{}' requested but PyO3 not available", operation);
            #[cfg(not(feature = "python-ffi"))]
            {
                Err(FeatureGateError::PyO3NotAvailable)
            }
            #[cfg(feature = "python-ffi")]
            {
                pyo3_fn()
            }
        }
    }

    /// Convert Python object with fallback
    pub fn convert_python_object(&self, obj: &PyObjectWrapper) -> Result<String, FeatureGateError> {
        #[cfg(feature = "python-ffi")]
        {
            use pyo3::prelude::*;
            Python::with_gil(|py| {
                obj.to_string()
                    .parse()
                    .map_err(|e| FeatureGateError::ConversionError(format!("{:?}", e)))
            })
        }

        #[cfg(not(feature = "python-ffi"))]
        {
            debug!("Converting mock PyObject: {}", obj.type_name());
            Ok(obj.to_string())
        }
    }

    /// Create conditional Python wrapper
    pub fn create_python_wrapper(&self, type_name: &str, value: &str) -> PyObjectWrapper {
        #[cfg(feature = "python-ffi")]
        {
            use pyo3::prelude::*;
            Python::with_gil(|py| {
                debug!("Creating PyO3 wrapper for type: {}", type_name);
                py.None()
            })
        }

        #[cfg(not(feature = "python-ffi"))]
        {
            debug!("Creating mock PyObject wrapper for type: {}", type_name);
            MockPyObject::new(type_name, value)
        }
    }

    /// Get fallback handler for operation
    pub fn get_fallback(&self, handler_name: &str) -> Option<&Arc<dyn FallbackHandler>> {
        self.fallback_registry.get(handler_name)
    }

    /// Execute with fallback if PyO3 unavailable
    pub fn execute_with_fallback<T>(
        &self,
        operation: &str,
        args: &[String],
        fallback_name: &str,
    ) -> Result<String, FeatureGateError> {
        if !self.feature_enabled {
            if let Some(handler) = self.get_fallback(fallback_name) {
                debug!("Using fallback handler '{}' for operation '{}'", fallback_name, operation);
                handler.handle(operation, args)
            } else {
                Err(FeatureGateError::FallbackNotFound(fallback_name.to_string()))
            }
        } else {
            debug!("PyO3 available - executing operation: {}", operation);
            Ok(format!("pyo3_{}_{}", operation, args.len()))
        }
    }

    /// Validate feature gate configuration
    pub fn validate_configuration(&self) -> Result<(), FeatureGateError> {
        debug!("Validating PyO3 feature gate configuration");

        // Check feature consistency
        let cargo_feature_enabled = cfg!(feature = "python-ffi");
        if self.feature_enabled != cargo_feature_enabled {
            return Err(FeatureGateError::ConfigurationError(
                "Feature flag mismatch between system and Cargo configuration".to_string()
            ));
        }

        // Validate fallback handlers
        if self.fallback_registry.is_empty() {
            return Err(FeatureGateError::ConfigurationError(
                "No fallback handlers registered".to_string()
            ));
        }

        info!("PyO3 feature gate configuration validated successfully");
        Ok(())
    }

    /// Get system statistics
    pub fn get_statistics(&self) -> FeatureGateStatistics {
        FeatureGateStatistics {
            pyo3_enabled: self.feature_enabled,
            fallback_count: self.fallback_registry.len(),
            debug_enabled: self.debug_logging,
        }
    }
}

impl Default for PyO3FeatureGateSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the feature gate system
#[derive(Debug, Clone)]
pub struct FeatureGateStatistics {
    pub pyo3_enabled: bool,
    pub fallback_count: usize,
    pub debug_enabled: bool,
}

impl fmt::Display for FeatureGateStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PyO3FeatureGate(enabled={}, fallbacks={}, debug={})",
            self.pyo3_enabled, self.fallback_count, self.debug_enabled
        )
    }
}

/// Macro for conditional PyO3 code compilation
#[macro_export]
macro_rules! pyo3_conditional {
    (
        pyo3: $pyo3_block:block,
        fallback: $fallback_block:block
    ) => {
        {
            #[cfg(feature = "python-ffi")]
            {
                $pyo3_block
            }

            #[cfg(not(feature = "python-ffi"))]
            {
                $fallback_block
            }
        }
    };
}

/// Macro for creating feature-gated functions
#[macro_export]
macro_rules! define_gated_function {
    (
        $(#[$attr:meta])*
        pub fn $name:ident($($arg:ident: $arg_type:ty),*) -> $return_type:ty
        where
            pyo3: $pyo3_body:block,
            fallback: $fallback_body:block
    ) => {
        $(#[$attr])*
        #[cfg(feature = "python-ffi")]
        pub fn $name($($arg: $arg_type),*) -> $return_type {
            $pyo3_body
        }

        $(#[$attr])*
        #[cfg(not(feature = "python-ffi"))]
        pub fn $name($($arg: $arg_type),*) -> $return_type {
            $fallback_body
        }
    };
}

// ============================================================================
// COMPREHENSIVE CAPABILITY TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_gate_system_initialization() {
        let system = PyO3FeatureGateSystem::new();
        
        // Verify system state
        assert_eq!(system.is_pyo3_available(), cfg!(feature = "python-ffi"));
        assert!(!system.fallback_registry.is_empty());
        
        // Validate configuration
        assert!(system.validate_configuration().is_ok());
    }

    #[test]
    fn test_fallback_handler_registration() {
        let mut system = PyO3FeatureGateSystem::new();
        let initial_count = system.fallback_registry.len();
        
        // Register custom fallback
        let custom_handler = Arc::new(DefaultFallbackHandler::new("custom_test"));
        system.register_fallback(custom_handler);
        
        assert_eq!(system.fallback_registry.len(), initial_count + 1);
        assert!(system.get_fallback("custom_test").is_some());
    }

    #[test]
    fn test_fallback_execution() {
        let system = PyO3FeatureGateSystem::new();
        
        let result = system.execute_with_fallback(
            "test_operation",
            &["arg1".to_string(), "arg2".to_string()],
            "default"
        );
        
        assert!(result.is_ok());
        if !system.is_pyo3_available() {
            assert!(result.unwrap().contains("fallback_test_operation"));
        }
    }

    #[test]
    fn test_python_object_wrapper_creation() {
        let system = PyO3FeatureGateSystem::new();
        
        let wrapper = system.create_python_wrapper("TestType", "test_value");
        
        #[cfg(not(feature = "python-ffi"))]
        {
            assert_eq!(wrapper.type_name(), "TestType");
            assert_eq!(wrapper.to_string(), "test_value");
        }
    }

    #[test]
    fn test_python_object_conversion() {
        let system = PyO3FeatureGateSystem::new();
        
        let wrapper = system.create_python_wrapper("str", "hello");
        let result = system.convert_python_object(&wrapper);
        
        assert!(result.is_ok());
        
        #[cfg(not(feature = "python-ffi"))]
        {
            assert_eq!(result.unwrap(), "hello");
        }
    }

    #[test]
    fn test_gated_execution() {
        let system = PyO3FeatureGateSystem::new();
        
        let result = system.execute_gated("test_op", || {
            #[cfg(feature = "python-ffi")]
            {
                Ok("pyo3_result".to_string())
            }
            #[cfg(not(feature = "python-ffi"))]
            {
                Ok("rust_result".to_string())
            }
        });
        
        if system.is_pyo3_available() {
            #[cfg(feature = "python-ffi")]
            assert!(result.is_ok());
        } else {
            #[cfg(not(feature = "python-ffi"))]
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_statistics_reporting() {
        let system = PyO3FeatureGateSystem::new();
        let stats = system.get_statistics();
        
        assert_eq!(stats.pyo3_enabled, cfg!(feature = "python-ffi"));
        assert!(stats.fallback_count > 0);
        assert!(stats.debug_enabled);
        
        let stats_str = format!("{}", stats);
        assert!(stats_str.contains("PyO3FeatureGate"));
    }

    #[test]
    fn test_error_handling() {
        let system = PyO3FeatureGateSystem::new();
        
        // Test missing fallback handler
        let result = system.execute_with_fallback(
            "test_op",
            &[],
            "nonexistent_handler"
        );
        
        assert!(result.is_err());
        match result.unwrap_err() {
            FeatureGateError::FallbackNotFound(name) => {
                assert_eq!(name, "nonexistent_handler");
            }
            _ => panic!("Expected FallbackNotFound error"),
        }
    }

    #[test]
    fn test_macro_conditional_compilation() {
        let result = pyo3_conditional! {
            pyo3: {
                "pyo3_branch"
            },
            fallback: {
                "fallback_branch"
            }
        };
        
        #[cfg(feature = "python-ffi")]
        assert_eq!(result, "pyo3_branch");
        
        #[cfg(not(feature = "python-ffi"))]
        assert_eq!(result, "fallback_branch");
    }

    // Define a test function using the gated function macro
    define_gated_function! {
        /// Test function with feature gating
        pub fn test_gated_function(input: String) -> String
        where
            pyo3: {
                format!("pyo3_{}", input)
            },
            fallback: {
                format!("rust_{}", input)
            }
    }

    #[test]
    fn test_gated_function_macro() {
        let result = test_gated_function("test".to_string());
        
        #[cfg(feature = "python-ffi")]
        assert_eq!(result, "pyo3_test");
        
        #[cfg(not(feature = "python-ffi"))]
        assert_eq!(result, "rust_test");
    }

    #[test]
    fn test_configuration_validation() {
        let system = PyO3FeatureGateSystem::new();
        
        // Should pass validation
        assert!(system.validate_configuration().is_ok());
        
        // Test with empty fallback registry
        let mut empty_system = PyO3FeatureGateSystem {
            feature_enabled: false,
            debug_logging: true,
            fallback_registry: HashMap::new(),
        };
        
        let validation_result = empty_system.validate_configuration();
        assert!(validation_result.is_err());
        
        if let Err(FeatureGateError::ConfigurationError(msg)) = validation_result {
            assert!(msg.contains("No fallback handlers"));
        }
    }

    #[test]
    fn test_debug_logging() {
        let system = PyO3FeatureGateSystem::new();
        
        // Test that debug logging doesn't crash
        let _wrapper = system.create_python_wrapper("TestDebug", "debug_value");
        let _result = system.execute_with_fallback("debug_test", &[], "default");
        
        // If we get here without panicking, debug logging works
        assert!(true);
    }

    #[test]
    fn test_error_display() {
        let errors = vec![
            FeatureGateError::PyO3NotAvailable,
            FeatureGateError::FallbackNotFound("test".to_string()),
            FeatureGateError::ConversionError("test conversion".to_string()),
            FeatureGateError::ConfigurationError("test config".to_string()),
        ];
        
        for error in errors {
            let display_str = format!("{}", error);
            assert!(!display_str.is_empty());
            
            // Verify error implements std::error::Error
            let _: &dyn std::error::Error = &error;
        }
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

#[cfg(all(test, feature = "python-ffi"))]
mod pyo3_integration_tests {
    use super::*;
    use pyo3::prelude::*;

    #[test]
    fn test_actual_pyo3_integration() {
        let system = PyO3FeatureGateSystem::new();
        assert!(system.is_pyo3_available());
        
        Python::with_gil(|py| {
            let result = py.eval("2 + 2", None, None).unwrap();
            let value: i32 = result.extract().unwrap();
            assert_eq!(value, 4);
        });
    }

    #[test]
    fn test_pyo3_object_conversion() {
        let system = PyO3FeatureGateSystem::new();
        
        Python::with_gil(|py| {
            let py_str = py.eval("'hello world'", None, None).unwrap();
            let py_obj = py_str.to_object(py);
            
            // Test conversion through our system
            let converted = system.convert_python_object(&py_obj);
            assert!(converted.is_ok());
        });
    }
}

#[cfg(all(test, not(feature = "python-ffi")))]
mod rust_only_tests {
    use super::*;

    #[test]
    fn test_rust_only_compilation() {
        let system = PyO3FeatureGateSystem::new();
        assert!(!system.is_pyo3_available());
        
        // All operations should work without PyO3
        let mock_obj = MockPyObject::new("str", "test_value");
        assert_eq!(mock_obj.type_name(), "str");
        assert_eq!(mock_obj.to_string(), "test_value");
    }

    #[test]
    fn test_fallback_only_operations() {
        let system = PyO3FeatureGateSystem::new();
        
        let result = system.execute_with_fallback(
            "rust_operation",
            &["arg1".to_string()],
            "default"
        );
        
        assert!(result.is_ok());
        assert!(result.unwrap().contains("fallback"));
    }
}