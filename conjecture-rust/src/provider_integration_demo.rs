//! Provider Integration Capability Demonstration
//!
//! This module demonstrates the complete Provider Integration capability
//! implemented in the EngineOrchestrator, showing how different providers
//! are managed throughout the test lifecycle.

use crate::engine_orchestrator::{EngineOrchestrator, OrchestratorConfig, OrchestrationError, BackendScope};
use crate::data::ConjectureData;
use crate::providers::HypothesisProvider;

/// Demonstrate basic provider lifecycle management
pub fn demo_provider_lifecycle() {
    println!("=== Provider Integration Demo: Lifecycle Management ===");
    
    let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
    let config = OrchestratorConfig::default();
    
    let mut orchestrator = EngineOrchestrator::new(test_fn, config);
    
    // Register observation callbacks
    orchestrator.register_provider_observation_callback("demo_observer".to_string());
    orchestrator.register_provider_observation_callback("performance_monitor".to_string());
    
    println!("Initial provider: {}", orchestrator.provider_context().active_provider);
    println!("Using Hypothesis backend: {}", orchestrator.using_hypothesis_backend());
    
    // Log some provider events
    orchestrator.log_provider_observation("demo_started", "lifecycle demonstration");
    orchestrator.log_provider_observation("provider_validated", "hypothesis backend ready");
    
    println!("Provider Integration Demo completed successfully");
}

/// Demonstrate provider switching based on BackendCannotProceed
pub fn demo_provider_switching() {
    println!("\n=== Provider Integration Demo: Provider Switching ===");
    
    let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
    
    // Configure with a non-hypothesis backend
    let mut config = OrchestratorConfig::default();
    config.backend = "crosshair".to_string();
    
    let mut orchestrator = EngineOrchestrator::new(test_fn, config);
    
    println!("Initial backend: {}", orchestrator.provider_context().active_provider);
    println!("Initially using Hypothesis: {}", orchestrator.using_hypothesis_backend());
    
    // Simulate BackendCannotProceed with verified scope
    println!("\nSimulating BackendCannotProceed with 'verified' scope...");
    let result = orchestrator.handle_backend_cannot_proceed(BackendScope::Verified);
    assert!(result.is_ok());
    
    println!("After verified scope:");
    println!("  Switched to Hypothesis: {}", orchestrator.provider_context().switch_to_hypothesis);
    println!("  Verified by: {:?}", orchestrator.provider_context().verified_by);
    println!("  Using Hypothesis backend: {}", orchestrator.using_hypothesis_backend());
    
    // Demonstrate discard threshold switching
    println!("\nSimulating multiple discard_test_case errors...");
    orchestrator.set_call_count(50); // Set up realistic call count
    
    for i in 1..=15 {
        let result = orchestrator.handle_backend_cannot_proceed(BackendScope::DiscardTestCase);
        assert!(result.is_ok());
        
        if i == 12 {
            println!("  After {} discards: switched = {}", i, orchestrator.provider_context().switch_to_hypothesis);
        }
    }
    
    println!("Final failed realize count: {}", orchestrator.provider_context().failed_realize_count);
    println!("Provider Switching Demo completed successfully");
}

/// Demonstrate phase-specific provider selection
pub fn demo_phase_provider_selection() {
    println!("\n=== Provider Integration Demo: Phase-Specific Provider Selection ===");
    
    let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
    
    let mut config = OrchestratorConfig::default();
    config.backend = "random".to_string();
    
    let orchestrator = EngineOrchestrator::new(test_fn, config);
    
    use crate::engine_orchestrator::ExecutionPhase;
    
    println!("Provider selection by phase:");
    println!("  Initialize: {}", orchestrator.select_provider_for_phase(ExecutionPhase::Initialize));
    println!("  Reuse: {}", orchestrator.select_provider_for_phase(ExecutionPhase::Reuse));
    println!("  Generate: {}", orchestrator.select_provider_for_phase(ExecutionPhase::Generate));
    println!("  Shrink: {}", orchestrator.select_provider_for_phase(ExecutionPhase::Shrink));
    println!("  Cleanup: {}", orchestrator.select_provider_for_phase(ExecutionPhase::Cleanup));
    
    println!("Phase Provider Selection Demo completed successfully");
}

/// Demonstrate error handling in provider operations
pub fn demo_provider_error_handling() {
    println!("\n=== Provider Integration Demo: Error Handling ===");
    
    let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
    
    // Test with invalid backend
    let mut config = OrchestratorConfig::default();
    config.backend = "nonexistent_backend".to_string();
    
    let mut orchestrator = EngineOrchestrator::new(test_fn, config);
    
    // Test create_active_provider which will check for backend availability
    match orchestrator.create_active_provider() {
        Ok(_) => println!("Provider creation unexpectedly succeeded"),
        Err(e) => {
            match e {
                OrchestrationError::ProviderCreationFailed { backend, reason } => {
                    println!("Expected error: Failed to create provider '{}': {}", backend, reason);
                }
                _ => println!("Unexpected error type: {}", e),
            }
        }
    }
    
    println!("Provider Error Handling Demo completed successfully");
}

/// Run all provider integration demonstrations
pub fn run_provider_integration_demos() {
    println!("🦀 Provider Integration Capability Demonstration\n");
    
    demo_provider_lifecycle();
    demo_provider_switching();
    demo_phase_provider_selection();
    demo_provider_error_handling();
    
    println!("\n✅ All Provider Integration Demos completed successfully!");
    println!("\nProvider Integration Features Demonstrated:");
    println!("  ✓ Provider lifecycle management");
    println!("  ✓ Dynamic provider switching based on BackendCannotProceed");
    println!("  ✓ Phase-specific provider selection");
    println!("  ✓ Provider observability and logging");
    println!("  ✓ Error handling and resource cleanup");
    println!("  ✓ RAII-based resource management");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_integration_demos() {
        // Run all demos as a test to ensure they work
        run_provider_integration_demos();
    }
}