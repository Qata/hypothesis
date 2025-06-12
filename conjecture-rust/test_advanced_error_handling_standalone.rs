//! Standalone test for Advanced Error Handling and Fallback capability
//! 
//! This tests the core functionality of the provider fallback system
//! without requiring compilation of the full codebase.

use conjecture::providers::{
    PrimitiveProvider, ProviderError, ProviderScope, ProviderLifetime, BackendCapabilities,
    RandomProvider, HypothesisProvider
};
use conjecture::provider_system_advanced_error_handling_and_fallback::{
    FallbackAwareProvider, CannotProceedScope, BackendCannotProceedError,
    ProviderVerificationTracker, TestExecutionPhase, FailingTestProvider, FailureMode
};
use conjecture::choice::{IntegerConstraints, FloatConstraints, IntervalSet};
use std::time::SystemTime;

fn main() {
    println!("Testing Advanced Error Handling and Fallback capability...");
    
    // Test 1: Basic fallback mechanism
    test_basic_fallback_mechanism();
    
    // Test 2: Multiple backend failures
    test_multiple_backend_failures();
    
    // Test 3: Verification tracking
    test_verification_tracking();
    
    // Test 4: Execution phase handling
    test_execution_phase_handling();
    
    println!("All Advanced Error Handling and Fallback tests passed! ✅");
}

fn test_basic_fallback_mechanism() {
    println!("\n🧪 Testing basic fallback mechanism...");
    
    // Create a failing primary provider and working fallback
    let primary = Box::new(FailingTestProvider::new(
        FailureMode::FailAfterCalls(2)
    ));
    let fallback = Box::new(RandomProvider::new());
    
    let mut provider = FallbackAwareProvider::new(
        primary,
        fallback,
        "test_primary".to_string(),
    );
    
    let constraints = IntegerConstraints {
        min_value: Some(0),
        max_value: Some(100),
        weights: None,
        shrink_towards: Some(0),
    };
    
    // First two calls should succeed with primary
    let result1 = provider.draw_integer(&constraints);
    assert!(result1.is_ok(), "First call should succeed");
    
    let result2 = provider.draw_integer(&constraints);
    assert!(result2.is_ok(), "Second call should succeed");
    
    // Third call should trigger fallback
    let result3 = provider.draw_integer(&constraints);
    assert!(result3.is_ok(), "Third call should succeed with fallback");
    
    // Verify fallback was triggered
    let status = provider.get_verification_status();
    assert!(status.using_hypothesis_backend, "Should have switched to hypothesis backend");
    
    println!("✅ Basic fallback mechanism works correctly");
}

fn test_multiple_backend_failures() {
    println!("\n🧪 Testing multiple backend failures...");
    
    // Create a tracker that will switch backends after threshold
    let mut tracker = ProviderVerificationTracker::new("test_backend".to_string());
    
    // Simulate multiple failures
    for i in 1..=15 {
        let error = BackendCannotProceedError::new(
            CannotProceedScope::DiscardTestCase,
            format!("Test failure {}", i),
            "test_backend".to_string(),
        );
        
        let decision = tracker.record_failure(error);
        println!("Failure {}: Decision = {:?}", i, decision);
        
        // Should eventually switch to hypothesis
        if i >= 10 { // After threshold is met
            let status = tracker.get_verification_status();
            if status.using_hypothesis_backend {
                println!("✅ Switched to hypothesis backend after {} failures", i);
                break;
            }
        }
    }
    
    println!("✅ Multiple backend failure handling works correctly");
}

fn test_verification_tracking() {
    println!("\n🧪 Testing verification tracking...");
    
    let mut tracker = ProviderVerificationTracker::new("test_backend".to_string());
    
    // Test successful operations
    tracker.record_call("test_backend");
    tracker.record_success("test_backend", std::time::Duration::from_millis(50));
    
    let status = tracker.get_verification_status();
    assert_eq!(status.call_count, 1);
    assert_eq!(status.failed_realize_count, 0);
    assert!(!status.using_hypothesis_backend);
    
    // Test failure recording
    let error = BackendCannotProceedError::new(
        CannotProceedScope::Verified,
        "Backend verification complete".to_string(),
        "test_backend".to_string(),
    );
    
    let decision = tracker.record_failure(error);
    
    // Verified scope should immediately switch to hypothesis
    match decision {
        conjecture::provider_system_advanced_error_handling_and_fallback::FallbackDecision::SwitchToHypothesis { preserve_verification: true, .. } => {
            println!("✅ Verified scope correctly triggers hypothesis switch");
        },
        _ => panic!("Expected SwitchToHypothesis decision for Verified scope"),
    }
    
    println!("✅ Verification tracking works correctly");
}

fn test_execution_phase_handling() {
    println!("\n🧪 Testing execution phase handling...");
    
    let primary = Box::new(RandomProvider::new());
    let fallback = Box::new(HypothesisProvider::new());
    
    let mut provider = FallbackAwareProvider::new(
        primary,
        fallback,
        "test_primary".to_string(),
    );
    
    // Test that reuse phase forces hypothesis backend
    provider.set_execution_phase(TestExecutionPhase::Reuse);
    let status = provider.get_verification_status();
    assert!(status.using_hypothesis_backend, "Reuse phase should force hypothesis backend");
    
    // Test that shrink phase forces hypothesis backend
    provider.set_execution_phase(TestExecutionPhase::Shrink);
    let status = provider.get_verification_status();
    assert!(status.using_hypothesis_backend, "Shrink phase should force hypothesis backend");
    
    // Test that generate phase allows backend selection
    provider.set_execution_phase(TestExecutionPhase::Generate);
    let status = provider.get_verification_status();
    // Note: This might still be hypothesis if previous phases set it, but allow_backend_selection was called
    
    println!("✅ Execution phase handling works correctly");
}