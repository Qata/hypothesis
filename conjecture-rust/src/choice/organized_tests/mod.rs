//! Comprehensive test suite organization for choice system
//! 
//! This module organizes tests by category and functionality for better maintenance
//! and discoverability. Tests are split into logical groups:

pub mod unit_tests;           // Basic functionality tests for individual components
pub mod integration_tests;    // Cross-module integration tests  
pub mod parity_tests;        // Python compatibility verification tests
pub mod property_tests;      // Property-based testing for correctness
pub mod performance_tests;   // Benchmarking and stress tests
pub mod regression_tests;    // Tests for specific bug fixes and edge cases
pub mod verification_tests;  // Direct Python algorithm verification tests
// pub mod python_ffi_tests;    // FFI verification (disabled for main lib)

// Re-export all test modules for easy access
pub use unit_tests::*;
pub use integration_tests::*;
pub use parity_tests::*;
pub use property_tests::*;
pub use performance_tests::*;
pub use regression_tests::*;
pub use verification_tests::*;
// pub use python_ffi_tests::*;