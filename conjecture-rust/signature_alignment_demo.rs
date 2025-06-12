//! Demonstration of the Test Function Signature Alignment capability
//! 
//! This demo shows how the signature alignment system resolves type mismatches
//! between EngineOrchestrator expectations and ConjectureData operation return types.

use crate::data::{ConjectureData, DrawError};
use crate::engine_orchestrator::{EngineOrchestrator, OrchestratorConfig, OrchestrationResult};
use crate::engine_orchestrator_test_function_signature_alignment::{
    ToOrchestrationResult, ConjectureDataOrchestrationExt, DrawErrorConverter,
    get_alignment_stats, reset_alignment_stats
};

/// Demonstrates the complete Test Function Signature Alignment capability
pub fn main() {
    println!("=== Test Function Signature Alignment Capability Demo ===\n");
    
    // Demo 1: Error conversion system
    demo_error_conversion();
    
    // Demo 2: Extension trait usage
    demo_extension_traits();
    
    // Demo 3: Full orchestrator integration
    demo_orchestrator_integration();
    
    println!("=== Signature Alignment Demo Complete ===");
}

fn demo_error_conversion() {
    println!("1. Error Conversion System:");
    println!("   Converting DrawError -> OrchestrationError\n");
    
    let test_errors = vec![
        DrawError::Overrun,
        DrawError::InvalidRange,
        DrawError::InvalidProbability,
        DrawError::Frozen,
        DrawError::UnsatisfiedAssumption("test condition failed".to_string()),
        DrawError::StopTest(0x42),
        DrawError::PreviouslyUnseenBehaviour,
    ];
    
    for error in test_errors {
        let converted = DrawErrorConverter::convert(error.clone());
        let category = DrawErrorConverter::error_category(&error);
        let is_terminal = DrawErrorConverter::is_terminal_error(&error);
        
        println!("   {:?}", error);
        println!("     -> {:?}", converted);
        println!("     -> Category: {}, Terminal: {}\n", category, is_terminal);
    }
}

fn demo_extension_traits() {
    println!("2. Extension Trait Usage:");
    println!("   ConjectureDataOrchestrationExt methods\n");
    
    let mut data = ConjectureData::new(42);
    
    // Demonstrate extension trait methods
    println!("   Using orchestration extension methods:");
    
    match data.draw_integer_orchestration(1, 100) {
        Ok(value) => println!("     ✅ Integer: {}", value),
        Err(e) => println!("     ❌ Integer error: {}", e),
    }
    
    match data.draw_boolean_orchestration(0.7) {
        Ok(value) => println!("     ✅ Boolean: {}", value),
        Err(e) => println!("     ❌ Boolean error: {}", e),
    }
    
    match data.assume_orchestration(true, "demo assumption") {
        Ok(()) => println!("     ✅ Assumption: passed"),
        Err(e) => println!("     ❌ Assumption error: {}", e),
    }
    
    match data.target_orchestration(85.5, "demo_score") {
        Ok(()) => println!("     ✅ Target: recorded"),
        Err(e) => println!("     ❌ Target error: {}", e),
    }
    
    println!();
}

fn demo_orchestrator_integration() {
    println!("3. Orchestrator Integration:");
    println!("   Full signature alignment in action\n");
    
    // Reset stats for clean demonstration
    reset_alignment_stats();
    
    // Create test function with signature alignment
    let test_fn = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
        // Method 1: Using extension traits (recommended)
        let value1 = data.draw_integer_orchestration(1, 100)?;
        let value2 = data.draw_boolean_orchestration(0.5)?;
        
        // Method 2: Using manual conversion (also supported)
        let _value3 = data.draw_float().to_orchestration_result()?;
        
        // Test property-based logic
        if value1 > 50 && value2 {
            data.assume_orchestration(false, "demo property violation")?;
        }
        
        // Record metrics
        data.target_orchestration(value1 as f64 / 100.0, "normalized_value")?;
        
        Ok(())
    });
    
    let config = OrchestratorConfig {
        max_examples: 10,
        backend: "hypothesis".to_string(),
        debug_logging: true,
        ..Default::default()
    };
    
    let mut orchestrator = EngineOrchestrator::new(test_fn, config);
    
    println!("   Running orchestrator with signature alignment...");
    
    match orchestrator.run() {
        Ok(stats) => {
            println!("     ✅ Execution completed successfully");
            if let Some(reason) = &stats.stopped_because {
                println!("     📊 Stopped because: {}", reason);
            }
        }
        Err(e) => {
            println!("     ⚠️  Execution completed with: {}", e);
        }
    }
    
    // Check alignment health
    match orchestrator.check_signature_alignment_health() {
        Ok(()) => println!("     ✅ Signature alignment health: GOOD"),
        Err(e) => println!("     ⚠️  Signature alignment health: {}", e),
    }
    
    // Show alignment statistics
    let final_stats = get_alignment_stats();
    println!("\n   📈 Signature Alignment Statistics:");
    println!("     Total operations: {}", final_stats.total_operations);
    println!("     Successful: {}", final_stats.successful_operations);
    println!("     Success rate: {:.1}%", final_stats.success_rate() * 100.0);
    
    if !final_stats.error_conversions.is_empty() {
        println!("     Error conversions:");
        for (error_type, count) in &final_stats.error_conversions {
            println!("       {}: {}", error_type, count);
        }
    }
    
    println!("\n   📋 Full Alignment Report:");
    let report = orchestrator.get_signature_alignment_report();
    for line in report.lines().take(10) {
        println!("     {}", line);
    }
    
    println!();
}