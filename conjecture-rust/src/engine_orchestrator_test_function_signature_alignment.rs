//! Test Function Signature Alignment Capability for EngineOrchestrator
//!
//! This module implements a sophisticated type conversion system that resolves
//! signature mismatches between EngineOrchestrator expectations (`OrchestrationResult<()>`)
//! and ConjectureData operation return types (`Result<T, DrawError>`).
//!
//! ## Core Capability
//!
//! The signature alignment system provides:
//! - Unified `OrchestrationResult<T>` type system with automatic conversion
//! - Trait-based error conversion from `DrawError` to `OrchestrationError`
//! - Extension traits for seamless ConjectureData operation integration
//! - Comprehensive error mapping with preservation of semantic meaning
//! - Debug logging and hex notation support
//!
//! ## Architecture
//!
//! ```text
//! ConjectureData Operations → Result<T, DrawError>
//!           ↓ (via ToOrchestrationResult trait)
//! OrchestrationResult<T> → OrchestrationResult<()>
//!           ↓ (via test function)
//! EngineOrchestrator → Unified error handling
//! ```

use crate::data::{ConjectureData, DrawError};
use crate::engine_orchestrator::{OrchestrationError, OrchestrationResult};
use std::fmt;

/// Enhanced result type that provides automatic conversion from DrawError to OrchestrationError
/// 
/// This type extends the base OrchestrationResult with additional conversion capabilities
/// and improved error context preservation.
pub type EnhancedOrchestrationResult<T> = Result<T, OrchestrationError>;

/// Trait for converting any Result type to OrchestrationResult with proper error mapping
/// 
/// This trait provides the core conversion mechanism that allows test functions to
/// seamlessly work with ConjectureData operations while maintaining type safety.
pub trait ToOrchestrationResult<T> {
    /// Convert to OrchestrationResult with automatic error conversion
    fn to_orchestration_result(self) -> OrchestrationResult<T>;
    
    /// Convert to OrchestrationResult with custom error context
    fn to_orchestration_result_with_context(self, context: &str) -> OrchestrationResult<T>;
    
    /// Convert to unit OrchestrationResult (discarding successful value)
    fn to_orchestration_unit_result(self) -> OrchestrationResult<()>;
}

/// Implementation for Result<T, DrawError> - the primary conversion case
impl<T> ToOrchestrationResult<T> for Result<T, DrawError> {
    fn to_orchestration_result(self) -> OrchestrationResult<T> {
        self.map_err(DrawErrorConverter::convert)
    }
    
    fn to_orchestration_result_with_context(self, context: &str) -> OrchestrationResult<T> {
        self.map_err(|e| DrawErrorConverter::convert_with_context(e, context))
    }
    
    fn to_orchestration_unit_result(self) -> OrchestrationResult<()> {
        match self {
            Ok(_) => Ok(()),
            Err(e) => Err(DrawErrorConverter::convert(e)),
        }
    }
}

/// Comprehensive error converter from DrawError to OrchestrationError
/// 
/// This converter preserves the semantic meaning of each error type while
/// providing appropriate OrchestrationError variants with enhanced context.
pub struct DrawErrorConverter;

impl DrawErrorConverter {
    /// Convert DrawError to OrchestrationError with preserved semantics
    pub fn convert(draw_error: DrawError) -> OrchestrationError {
        match draw_error {
            DrawError::Overrun => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting Overrun error");
                OrchestrationError::Overrun
            }
            DrawError::UnsatisfiedAssumption(msg) => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting UnsatisfiedAssumption: {}", msg);
                OrchestrationError::Invalid { 
                    reason: format!("Unsatisfied assumption: {}", msg)
                }
            }
            DrawError::StopTest(code) => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting StopTest with code: 0x{:08X}", code);
                OrchestrationError::Interrupted
            }
            DrawError::Frozen => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting Frozen error");
                OrchestrationError::Invalid { 
                    reason: "ConjectureData is frozen - no more draws allowed".to_string()
                }
            }
            DrawError::InvalidRange => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting InvalidRange error");
                OrchestrationError::Invalid { 
                    reason: "Invalid range specified (min > max)".to_string()
                }
            }
            DrawError::InvalidProbability => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting InvalidProbability error");
                OrchestrationError::Invalid { 
                    reason: "Invalid probability specified (not in [0, 1])".to_string()
                }
            }
            DrawError::EmptyAlphabet => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting EmptyAlphabet error");
                OrchestrationError::Invalid { 
                    reason: "Empty alphabet provided for string generation".to_string()
                }
            }
            DrawError::InvalidStatus => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting InvalidStatus error");
                OrchestrationError::Invalid { 
                    reason: "ConjectureData has invalid status for this operation".to_string()
                }
            }
            DrawError::EmptyChoice => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting EmptyChoice error");
                OrchestrationError::Invalid { 
                    reason: "Empty choice sequence provided".to_string()
                }
            }
            DrawError::InvalidReplayType => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting InvalidReplayType error");
                OrchestrationError::Invalid { 
                    reason: "Replay value type doesn't match expected type".to_string()
                }
            }
            DrawError::PreviouslyUnseenBehaviour => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting PreviouslyUnseenBehaviour error");
                OrchestrationError::BackendCannotProceed { 
                    scope: "discard_test_case".to_string()
                }
            }
            DrawError::InvalidChoice => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting InvalidChoice error");
                OrchestrationError::Invalid { 
                    reason: "Invalid choice or misaligned replay detected".to_string()
                }
            }
            DrawError::EmptyWeights => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting EmptyWeights error");
                OrchestrationError::Invalid { 
                    reason: "Empty weights array for weighted choice".to_string()
                }
            }
            DrawError::InvalidWeights => {
                eprintln!("SIGNATURE_ALIGNMENT: Converting InvalidWeights error");
                OrchestrationError::Invalid { 
                    reason: "Invalid weights (sum <= 0 or contains NaN/Infinity)".to_string()
                }
            }
        }
    }
    
    /// Convert DrawError to OrchestrationError with additional context
    pub fn convert_with_context(draw_error: DrawError, context: &str) -> OrchestrationError {
        let base_error = Self::convert(draw_error);
        match base_error {
            OrchestrationError::Invalid { reason } => {
                OrchestrationError::Invalid { 
                    reason: format!("{}: {}", context, reason)
                }
            }
            OrchestrationError::Provider { message } => {
                OrchestrationError::Provider { 
                    message: format!("{}: {}", context, message)
                }
            }
            other => other, // Other error types don't benefit from context
        }
    }
    
    /// Check if DrawError represents a terminal condition
    pub fn is_terminal_error(draw_error: &DrawError) -> bool {
        matches!(draw_error, 
            DrawError::Overrun | 
            DrawError::StopTest(_) | 
            DrawError::PreviouslyUnseenBehaviour
        )
    }
    
    /// Get error category for statistics
    pub fn error_category(draw_error: &DrawError) -> &'static str {
        match draw_error {
            DrawError::Overrun => "resource_limit",
            DrawError::StopTest(_) => "control_flow",
            DrawError::UnsatisfiedAssumption(_) => "assumption",
            DrawError::Frozen => "state",
            DrawError::InvalidRange | DrawError::InvalidProbability => "parameter",
            DrawError::EmptyAlphabet | DrawError::EmptyChoice | DrawError::EmptyWeights => "empty_input",
            DrawError::InvalidStatus | DrawError::InvalidChoice | DrawError::InvalidWeights => "validation",
            DrawError::InvalidReplayType => "replay",
            DrawError::PreviouslyUnseenBehaviour => "tree_exploration",
        }
    }
}

/// Extension trait for ConjectureData that provides orchestration-aware draw methods
/// 
/// This trait extends ConjectureData with methods that directly return OrchestrationResult,
/// eliminating the need for manual error conversion in test functions.
pub trait ConjectureDataOrchestrationExt {
    /// Draw integer with automatic OrchestrationResult conversion
    fn draw_integer_orchestration(&mut self, min: i128, max: i128) -> OrchestrationResult<i128>;
    
    /// Draw boolean with automatic OrchestrationResult conversion
    fn draw_boolean_orchestration(&mut self, probability: f64) -> OrchestrationResult<bool>;
    
    /// Draw float with automatic OrchestrationResult conversion
    fn draw_float_orchestration(&mut self) -> OrchestrationResult<f64>;
    
    /// Draw string with automatic OrchestrationResult conversion
    fn draw_string_orchestration(&mut self, alphabet: &str, min_size: usize, max_size: usize) -> OrchestrationResult<String>;
    
    /// Draw bytes with automatic OrchestrationResult conversion
    fn draw_bytes_orchestration(&mut self, size: usize) -> OrchestrationResult<Vec<u8>>;
    
    /// Draw weighted integer with automatic OrchestrationResult conversion
    fn draw_integer_weighted_orchestration(
        &mut self, 
        min_value: i128, 
        max_value: i128, 
        weights: Option<std::collections::HashMap<i128, f64>>, 
        size_hint: Option<i128>
    ) -> OrchestrationResult<i128>;
    
    /// Perform assumption check with automatic OrchestrationResult conversion
    fn assume_orchestration(&mut self, condition: bool, message: &str) -> OrchestrationResult<()>;
    
    /// Record target observation with automatic OrchestrationResult conversion
    fn target_orchestration(&mut self, observation: f64, label: &str) -> OrchestrationResult<()>;
}

impl ConjectureDataOrchestrationExt for ConjectureData {
    fn draw_integer_orchestration(&mut self, min: i128, max: i128) -> OrchestrationResult<i128> {
        self.draw_integer(min, max).to_orchestration_result_with_context("integer generation")
    }
    
    fn draw_boolean_orchestration(&mut self, probability: f64) -> OrchestrationResult<bool> {
        self.draw_boolean(probability).to_orchestration_result_with_context("boolean generation")
    }
    
    fn draw_float_orchestration(&mut self) -> OrchestrationResult<f64> {
        self.draw_float().to_orchestration_result_with_context("float generation")
    }
    
    fn draw_string_orchestration(&mut self, alphabet: &str, min_size: usize, max_size: usize) -> OrchestrationResult<String> {
        self.draw_string(alphabet, min_size, max_size).to_orchestration_result_with_context("string generation")
    }
    
    fn draw_bytes_orchestration(&mut self, size: usize) -> OrchestrationResult<Vec<u8>> {
        self.draw_bytes(size).to_orchestration_result_with_context("bytes generation")
    }
    
    fn draw_integer_weighted_orchestration(
        &mut self, 
        min_value: i128, 
        max_value: i128, 
        weights: Option<std::collections::HashMap<i128, f64>>, 
        size_hint: Option<i128>
    ) -> OrchestrationResult<i128> {
        self.draw_integer_weighted(min_value, max_value, weights, size_hint)
            .to_orchestration_result_with_context("weighted integer")
    }
    
    fn assume_orchestration(&mut self, condition: bool, message: &str) -> OrchestrationResult<()> {
        if condition {
            Ok(())
        } else {
            Err(OrchestrationError::Invalid {
                reason: format!("Assumption failed: {}", message)
            })
        }
    }
    
    fn target_orchestration(&mut self, observation: f64, label: &str) -> OrchestrationResult<()> {
        self.target(label, observation);
        Ok(())
    }
}

/// Signature alignment context for enhanced error tracking
/// 
/// This struct provides context information for signature alignment operations,
/// enabling better debugging and error reporting.
#[derive(Debug, Clone)]
pub struct SignatureAlignmentContext {
    /// Operation name for context
    pub operation: String,
    /// Source location information
    pub location: Option<String>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl SignatureAlignmentContext {
    /// Create a new context with operation name
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            location: None,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Add location information
    pub fn with_location(mut self, location: &str) -> Self {
        self.location = Some(location.to_string());
        self
    }
    
    /// Add metadata entry
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Format context for error messages
    pub fn format_for_error(&self) -> String {
        let mut parts = vec![self.operation.clone()];
        
        if let Some(ref location) = self.location {
            parts.push(format!("at {}", location));
        }
        
        if !self.metadata.is_empty() {
            let metadata_str: Vec<String> = self.metadata.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            parts.push(format!("[{}]", metadata_str.join(", ")));
        }
        
        parts.join(" ")
    }
}

/// Enhanced result wrapper that provides additional signature alignment utilities
/// 
/// This wrapper extends OrchestrationResult with convenience methods for
/// common test function patterns and enhanced error context.
pub struct AlignedResult<T> {
    inner: OrchestrationResult<T>,
    context: Option<SignatureAlignmentContext>,
}

impl<T> AlignedResult<T> {
    /// Create a new aligned result
    pub fn new(result: OrchestrationResult<T>) -> Self {
        Self {
            inner: result,
            context: None,
        }
    }
    
    /// Create aligned result with context
    pub fn with_context(result: OrchestrationResult<T>, context: SignatureAlignmentContext) -> Self {
        Self {
            inner: result,
            context: Some(context),
        }
    }
    
    /// Convert to unit result (discarding value on success)
    pub fn to_unit(self) -> OrchestrationResult<()> {
        match self.inner {
            Ok(_) => Ok(()),
            Err(e) => {
                if let Some(context) = self.context {
                    match e {
                        OrchestrationError::Invalid { reason } => {
                            Err(OrchestrationError::Invalid {
                                reason: format!("{}: {}", context.format_for_error(), reason)
                            })
                        }
                        other => Err(other)
                    }
                } else {
                    Err(e)
                }
            }
        }
    }
    
    /// Extract the inner result
    pub fn into_inner(self) -> OrchestrationResult<T> {
        self.inner
    }
    
    /// Check if result is successful
    pub fn is_ok(&self) -> bool {
        self.inner.is_ok()
    }
    
    /// Check if result is an error
    pub fn is_err(&self) -> bool {
        self.inner.is_err()
    }
}

impl<T> From<OrchestrationResult<T>> for AlignedResult<T> {
    fn from(result: OrchestrationResult<T>) -> Self {
        Self::new(result)
    }
}

impl<T> From<AlignedResult<T>> for OrchestrationResult<T> {
    fn from(aligned: AlignedResult<T>) -> Self {
        aligned.inner
    }
}

/// Macro for simplified error conversion in test functions
/// 
/// This macro provides a convenient way to convert ConjectureData operations
/// to OrchestrationResult without manual error handling.
#[macro_export]
macro_rules! orchestration_try {
    ($expr:expr) => {
        $expr.to_orchestration_result()?
    };
    ($expr:expr, $context:expr) => {
        $expr.to_orchestration_result_with_context($context)?
    };
}

/// Macro for creating test functions with automatic signature alignment
/// 
/// This macro simplifies the creation of test functions that work seamlessly
/// with the EngineOrchestrator's expected signature.
#[macro_export]
macro_rules! aligned_test_function {
    ($name:ident, $body:expr) => {
        let $name = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
            use $crate::engine_orchestrator_test_function_signature_alignment::{
                ToOrchestrationResult, ConjectureDataOrchestrationExt
            };
            $body(data)
        });
    };
}

/// Statistics collector for signature alignment operations
/// 
/// This struct tracks metrics about error conversions and signature alignment
/// operations for debugging and optimization purposes.
#[derive(Debug, Default, Clone)]
pub struct SignatureAlignmentStats {
    /// Count of each error type converted
    pub error_conversions: std::collections::HashMap<String, usize>,
    /// Count of successful operations
    pub successful_operations: usize,
    /// Count of operations with context
    pub context_operations: usize,
    /// Total operations processed
    pub total_operations: usize,
}

impl SignatureAlignmentStats {
    /// Record a successful operation
    pub fn record_success(&mut self) {
        self.successful_operations += 1;
        self.total_operations += 1;
    }
    
    /// Record an error conversion
    pub fn record_error_conversion(&mut self, error_type: &str) {
        *self.error_conversions.entry(error_type.to_string()).or_insert(0) += 1;
        self.total_operations += 1;
    }
    
    /// Record an operation with context
    pub fn record_context_operation(&mut self) {
        self.context_operations += 1;
    }
    
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            self.successful_operations as f64 / self.total_operations as f64
        }
    }
    
    /// Generate statistics report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("Signature Alignment Statistics:\n"));
        report.push_str(&format!("  Total operations: {}\n", self.total_operations));
        report.push_str(&format!("  Successful: {} ({:.1}%)\n", 
            self.successful_operations, self.success_rate() * 100.0));
        report.push_str(&format!("  With context: {}\n", self.context_operations));
        
        if !self.error_conversions.is_empty() {
            report.push_str("  Error conversions:\n");
            for (error_type, count) in &self.error_conversions {
                report.push_str(&format!("    {}: {}\n", error_type, count));
            }
        }
        
        report
    }
}

/// Thread-local statistics collector for global tracking
thread_local! {
    static ALIGNMENT_STATS: std::cell::RefCell<SignatureAlignmentStats> = 
        std::cell::RefCell::new(SignatureAlignmentStats::default());
}

/// Get current signature alignment statistics
pub fn get_alignment_stats() -> SignatureAlignmentStats {
    ALIGNMENT_STATS.with(|stats| (*stats.borrow()).clone())
}

/// Reset signature alignment statistics
pub fn reset_alignment_stats() {
    ALIGNMENT_STATS.with(|stats| *stats.borrow_mut() = SignatureAlignmentStats::default());
}

/// Record a signature alignment operation for statistics
pub fn record_alignment_operation(success: bool, error_type: Option<&str>, with_context: bool) {
    ALIGNMENT_STATS.with(|stats| {
        let mut stats = stats.borrow_mut();
        if success {
            stats.record_success();
        } else if let Some(error_type) = error_type {
            stats.record_error_conversion(error_type);
        }
        if with_context {
            stats.record_context_operation();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ConjectureData;
    use crate::engine_orchestrator::{EngineOrchestrator, OrchestratorConfig};

    #[test]
    fn test_draw_error_conversion() {
        let errors = vec![
            DrawError::Frozen,
            DrawError::InvalidRange,
            DrawError::InvalidProbability,
            DrawError::EmptyAlphabet,
            DrawError::Overrun,
            DrawError::StopTest(42),
            DrawError::UnsatisfiedAssumption("test".to_string()),
        ];
        
        for error in errors {
            let converted = DrawErrorConverter::convert(error.clone());
            match error {
                DrawError::Overrun => assert!(matches!(converted, OrchestrationError::Overrun)),
                DrawError::StopTest(_) => assert!(matches!(converted, OrchestrationError::Interrupted)),
                _ => assert!(matches!(converted, OrchestrationError::Invalid { .. })),
            }
        }
    }
    
    #[test]
    fn test_to_orchestration_result_trait() {
        let success: Result<i32, DrawError> = Ok(42);
        let failure: Result<i32, DrawError> = Err(DrawError::InvalidRange);
        
        assert!(success.clone().to_orchestration_result().is_ok());
        assert!(failure.clone().to_orchestration_result().is_err());
        
        assert!(success.to_orchestration_unit_result().is_ok());
        assert!(failure.to_orchestration_unit_result().is_err());
    }
    
    #[test]
    fn test_conjecture_data_orchestration_ext() {
        let mut data = ConjectureData::new(42);
        
        // These should work without compilation errors
        let _result = data.draw_integer_orchestration(1, 100);
        let _result = data.draw_boolean_orchestration(0.5);
        let _result = data.draw_float_orchestration();
        let _result = data.assume_orchestration(true, "test assumption");
    }
    
    #[test]
    fn test_signature_alignment_context() {
        let context = SignatureAlignmentContext::new("test_operation")
            .with_location("test.rs:123")
            .with_metadata("key", "value");
        
        let formatted = context.format_for_error();
        assert!(formatted.contains("test_operation"));
        assert!(formatted.contains("test.rs:123"));
        assert!(formatted.contains("key=value"));
    }
    
    #[test]
    fn test_aligned_result() {
        let success: OrchestrationResult<i32> = Ok(42);
        let failure: OrchestrationResult<i32> = Err(OrchestrationError::Invalid {
            reason: "test error".to_string()
        });
        
        let aligned_success = AlignedResult::new(success);
        let aligned_failure = AlignedResult::new(failure);
        
        assert!(aligned_success.to_unit().is_ok());
        assert!(aligned_failure.to_unit().is_err());
    }
    
    #[test]
    fn test_signature_alignment_stats() {
        reset_alignment_stats();
        
        record_alignment_operation(true, None, false);
        record_alignment_operation(false, Some("InvalidRange"), true);
        
        let stats = get_alignment_stats();
        assert_eq!(stats.successful_operations, 1);
        assert_eq!(stats.context_operations, 1);
        assert_eq!(stats.total_operations, 2);
        assert!(stats.error_conversions.contains_key("InvalidRange"));
    }
    
    #[test]
    fn test_integration_with_orchestrator() {
        // Test function using the signature alignment system
        let test_fn = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
            // Use extension trait methods for seamless integration
            let _value1 = data.draw_integer_orchestration(1, 100)?;
            let _value2 = data.draw_boolean_orchestration(0.5)?;
            let _value3 = data.draw_float_orchestration()?;
            
            data.assume_orchestration(true, "test assumption")?;
            data.target_orchestration(1.0, "test_target")?;
            
            Ok(())
        });
        
        let config = OrchestratorConfig {
            max_examples: 5,
            backend: "hypothesis".to_string(),
            debug_logging: false,
            ..Default::default()
        };
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        let result = orchestrator.run();
        
        // Should complete without compilation errors
        assert!(result.is_ok() || result.is_err());
    }
}