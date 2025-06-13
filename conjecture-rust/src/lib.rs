//! # Conjecture Rust 2: Electric Boogaloo
//!
//! A faithful port of Python Hypothesis's modern conjecture engine architecture to Rust.
//! 
//! This implementation follows Python's choice-based design where all randomness flows
//! through strongly-typed choices with associated constraints.

pub mod choice;
pub mod data;
pub mod data_helper;
pub mod datatree;
pub mod datatree_enhanced_navigation;
pub mod choice_sequence_management;
pub mod choice_sequence_integration;
pub mod shrinking;
pub mod engine;
pub mod engine_orchestrator;
pub mod engine_orchestrator_provider_type_integration;
pub mod engine_orchestrator_choice_system_shrinking_integration;
pub mod engine_orchestrator_test_function_signature_alignment;
pub mod engine_orchestrator_datatree_novel_prefix_integration;

#[cfg(test)]
mod engine_orchestrator_choice_system_shrinking_integration_test;
#[cfg(test)]
mod engine_orchestrator_choice_system_shrinking_integration_comprehensive_capability_tests;
pub mod conjecture_data_lifecycle_management;
pub mod conjecture_data_lifecycle_comprehensive_capability_tests;
pub mod provider_integration_demo;
pub mod providers;
pub mod provider_lifecycle_management;
pub mod persistence;
pub mod targeting;
pub mod float_performance_test;
pub mod float_encoding_export;
pub mod pyo3_feature_gate_system_comprehensive_capability_tests;

#[cfg(test)]
mod provider_verification_test;
#[cfg(test)]
pub mod provider_system_coverage_guided_generation_integration_comprehensive_capability_tests;
#[cfg(test)]
pub mod provider_lifecycle_management_comprehensive_capability_tests;
#[cfg(test)]
pub mod provider_system_template_based_generation_comprehensive_capability_tests;
pub mod provider_system_advanced_error_handling_and_fallback;
#[cfg(test)]
pub mod provider_system_advanced_error_handling_and_fallback_comprehensive_capability_tests;
#[cfg(all(test, feature = "python-ffi"))]
pub mod provider_lifecycle_management_comprehensive_pyo3_ffi_integration_tests;

// Python FFI Integration modules (conditionally compiled)
#[cfg(feature = "python-ffi")]
pub mod conjecture_data_python_ffi;
#[cfg(feature = "python-ffi")]
pub mod conjecture_data_python_ffi_advanced;
#[cfg(feature = "python-ffi")]
pub mod conjecture_data_python_ffi_validation_tests;
#[cfg(feature = "python-ffi")]
pub mod conjecture_data_python_ffi_integration;

// Re-export core types for easy access
pub use choice::{ChoiceNode, ChoiceType, ChoiceValue, Constraints, FloatConstraintTypeSystem, FloatGenerationStrategy, FloatConstraintAwareProvider};
pub use data::{ConjectureData, ConjectureResult, Example, Status, DrawError, DataObserver, TreeRecordingObserver};
pub use engine_orchestrator_test_function_signature_alignment::{
    ToOrchestrationResult, ConjectureDataOrchestrationExt, DrawErrorConverter, 
    SignatureAlignmentContext, AlignedResult, EnhancedOrchestrationResult
};
pub use choice_sequence_management::{
    ChoiceSequenceManager, EnhancedChoiceNode, ChoiceSequenceError, 
    SequenceIntegrityStatus, PerformanceMetrics, TypeMetadata, ConstraintMetadata, ReplayMetadata
};
pub use datatree::{DataTree, TreeNode, TreeStats, Transition};
pub use datatree_enhanced_navigation::{TreeRecordingObserver as EnhancedTreeRecordingObserver, NavigationState, NavigationStats, ChildSelectionStrategy};
pub use shrinking::{ChoiceShrinker, ShrinkingTransformation};
pub use engine::{ConjectureRunner, RunnerConfig, RunnerStats, RunResult};
pub use engine_orchestrator::{EngineOrchestrator, OrchestratorConfig, ExecutionPhase, ExecutionStatistics, OrchestrationError};
pub use engine_orchestrator_choice_system_shrinking_integration::{
    ChoiceSystemShrinkingIntegration, ShrinkingIntegrationConfig, ShrinkingIntegrationResult, 
    AdvancedShrinkResult, ConversionError
};
pub use engine_orchestrator_datatree_novel_prefix_integration::{
    NovelPrefixGenerator, NovelPrefixIntegrationConfig, NovelPrefixIntegrationStats,
    NovelPrefixGenerationResult, SimulationResult
};
pub use providers::{PrimitiveProvider, HypothesisProvider, RandomProvider, ProviderRegistry, get_provider_registry};
pub use provider_lifecycle_management::{
    ProviderLifecycleManager, ManagedProvider, LifecycleScope, LifecycleEvent, LifecycleHooks,
    DefaultLifecycleHooks, ProviderInstanceMetadata, ProviderState, ProviderMetrics,
    CacheConfiguration, CleanupTask, CleanupTaskType
};
pub use persistence::{ExampleDatabase, DatabaseKey, DirectoryDatabase, InMemoryDatabase, DatabaseError, DatabaseIntegration, ExampleSerialization};
// pub use targeting::{TargetingEngine, TargetFunction, TargetObservation, ParetoPoint, CoverageState, OptimizationDirection, MinimizeFunction, MaximizeFunction, ComplexityFunction, TargetingSuggestions};
pub use float_encoding_export::{
    float_to_lex, lex_to_float, float_to_int, int_to_float,
    FloatWidth, FloatEncodingStrategy, FloatEncodingResult, FloatEncodingConfig, EncodingDebugInfo,
    build_exponent_tables, build_exponent_tables_for_width_export,
    float_to_lex_advanced, float_to_lex_multi_width, lex_to_float_multi_width
};

// Python FFI Integration re-exports (conditionally compiled)
#[cfg(feature = "python-ffi")]
pub use conjecture_data_python_ffi::{
    FfiError, ConstraintPythonSerializable, ConstraintPythonDeserializable
};
#[cfg(feature = "python-ffi")]
pub use conjecture_data_python_ffi_advanced::{
    ChoiceSequenceBinaryCodec, ConstraintValidator, BulkOperations, StateManager
};
#[cfg(feature = "python-ffi")]
pub use conjecture_data_python_ffi_validation_tests::{
    ConjectureDataValidationSuite
};
#[cfg(feature = "python-ffi")]
pub use conjecture_data_python_ffi_integration::{
    ConjectureDataPythonIntegration
};

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_test() {
        // This test will be replaced when we port Python tests
        // Just ensuring the project compiles for now
        assert_eq!(2 + 2, 4);
    }
}

#[cfg(test)]
mod tdd_verification;

#[cfg(test)]
mod shrinking_parity_tests;

#[cfg(test)]
mod python_interop_tests;

#[cfg(test)]
mod datatree_integration_tests;

#[cfg(test)]
mod status_tests;

#[cfg(test)]
mod status_integration_test;

#[cfg(test)]
mod status_verification_test;

#[cfg(test)]
mod buffer_operations_test;

// #[cfg(test)]
// mod engine_orchestrator_test_function_signature_alignment_tests;

// #[cfg(test)]
// mod targeting_comprehensive_capability_tests;