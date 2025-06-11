//! # Conjecture Rust 2: Electric Boogaloo
//!
//! A faithful port of Python Hypothesis's modern conjecture engine architecture to Rust.
//! 
//! This implementation follows Python's choice-based design where all randomness flows
//! through strongly-typed choices with associated constraints.

pub mod choice;
pub mod data;
pub mod datatree;
pub mod shrinking;
pub mod engine;
pub mod providers;
pub mod float_performance_test;

// Re-export core types for easy access
pub use choice::{ChoiceNode, ChoiceType, ChoiceValue, Constraints};
pub use data::{ConjectureData, ConjectureResult, Example, Status, DrawError, DataObserver, TreeRecordingObserver};
pub use datatree::{DataTree, TreeNode, TreeStats, Transition};
pub use shrinking::{ChoiceShrinker, ShrinkingTransformation};
pub use engine::{ConjectureRunner, RunnerConfig, RunnerStats, RunResult};
pub use providers::{PrimitiveProvider, HypothesisProvider, RandomProvider, ProviderRegistry, get_provider_registry};

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