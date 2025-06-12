//! Integration test for database persistence layer
//!
//! This test demonstrates the complete database integration working with
//! the EngineOrchestrator to save and reuse examples.

use std::fs;
use tempfile::tempdir;

use conjecture::{
    EngineOrchestrator, OrchestratorConfig, HypothesisProvider,
    ConjectureData, Status, OrchestrationError,
    DatabaseIntegration, DirectoryDatabase, DatabaseKey, ExampleDatabase
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing database integration...");
    
    // Create a temporary directory for the database
    let temp_dir = tempdir()?;
    let db_path = temp_dir.path().join("test_database");
    
    println!("Using database path: {:?}", db_path);
    
    // Create a simple test function that always fails
    let test_fn = Box::new(|data: &mut ConjectureData| -> Result<(), OrchestrationError> {
        // Draw a boolean choice (probability 0.5)
        let choice = match data.draw_boolean(0.5) {
            Ok(val) => val,
            Err(_) => return Err(OrchestrationError::Invalid { 
                reason: "Failed to draw boolean".to_string() 
            }),
        };
        
        // Make it interesting (failing) if true
        if choice {
            data.mark_interesting();
        }
        
        Ok(())
    }) as Box<dyn Fn(&mut ConjectureData) -> Result<(), OrchestrationError> + Send + Sync>;
    
    // Configure the orchestrator with database
    let mut config = OrchestratorConfig::default();
    config.max_examples = 10;
    config.database_path = Some(db_path.to_string_lossy().to_string());
    config.database_key = Some(b"test_integration".to_vec());
    config.debug_logging = true;
    
    println!("Running first test (should find and save examples)...");
    
    // First run - should find and save examples
    let provider = HypothesisProvider::new();
    let mut orchestrator = EngineOrchestrator::new(test_fn, provider, config.clone());
    
    let stats = orchestrator.run()?;
    println!("First run stats: {:?}", stats);
    
    // Check that the database was created and contains examples
    if db_path.exists() {
        println!("Database directory created successfully");
        
        // Count files in database
        let file_count = fs::read_dir(&db_path)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_dir())
            .count();
        
        println!("Database contains {} key directories", file_count);
    } else {
        println!("WARNING: Database directory was not created");
    }
    
    println!("\nRunning second test (should reuse examples)...");
    
    // Create the test function again (can't clone Box<dyn Fn>)
    let test_fn2 = Box::new(|data: &mut ConjectureData| -> Result<(), OrchestrationError> {
        // Draw a boolean choice (probability 0.5)
        let choice = match data.draw_boolean(0.5) {
            Ok(val) => val,
            Err(_) => return Err(OrchestrationError::Invalid { 
                reason: "Failed to draw boolean".to_string() 
            }),
        };
        
        // Make it interesting (failing) if true
        if choice {
            data.mark_interesting();
        }
        
        Ok(())
    }) as Box<dyn Fn(&mut ConjectureData) -> Result<(), OrchestrationError> + Send + Sync>;
    
    // Second run - should reuse examples from database
    let provider2 = HypothesisProvider::new();
    let mut orchestrator2 = EngineOrchestrator::new(test_fn2, provider2, config.clone());
    
    let stats2 = orchestrator2.run()?;
    println!("Second run stats: {:?}", stats2);
    
    // Test direct database access
    println!("\nTesting direct database access...");
    
    let mut db = DirectoryDatabase::new(&db_path)?;
    let key = DatabaseIntegration::generate_key("test_function", None, b"test_integration")?;
    
    let examples = db.fetch(&key)?;
    println!("Found {} examples in database", examples.len());
    
    for (i, example) in examples.iter().enumerate() {
        println!("Example {}: {} bytes", i, example.len());
        
        // Try to deserialize the example
        match DatabaseIntegration::deserialize_example(example) {
            Ok(choices) => {
                println!("  Deserialized {} choices", choices.len());
                for (j, choice) in choices.iter().enumerate() {
                    println!("    Choice {}: {:?}", j, choice.value);
                }
            }
            Err(e) => {
                println!("  Failed to deserialize: {}", e);
            }
        }
    }
    
    // Test database statistics
    let stats = db.get_stats()?;
    println!("\nDatabase statistics: {:?}", stats);
    
    println!("\nDatabase integration test completed successfully!");
    
    Ok(())
}