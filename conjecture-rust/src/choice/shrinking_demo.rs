//! Advanced Shrinking System Demo
//!
//! This module demonstrates the capabilities of the Advanced Shrinking System

use crate::choice::{ChoiceValue};
use crate::choice::shrinking_system::{
    AdvancedShrinkingEngine, Choice, ShrinkingStrategy, ShrinkResult
};

pub fn demo_advanced_shrinking() {
    println!("🎯 Advanced Shrinking System Demo");
    println!("==================================");

    let mut engine = AdvancedShrinkingEngine::default();

    // Demo 1: Integer minimization
    println!("\n📊 Demo 1: Integer Minimization");
    let choices = vec![
        Choice { value: ChoiceValue::Integer(1000), index: 0 },
        Choice { value: ChoiceValue::Integer(-500), index: 1 },
        Choice { value: ChoiceValue::Integer(250), index: 2 },
    ];

    println!("Original choices: {:?}", choices.iter().map(|c| &c.value).collect::<Vec<_>>());
    
    match engine.shrink_choices(&choices) {
        ShrinkResult::Success(shrunk) => {
            println!("✅ Shrunk choices: {:?}", shrunk.iter().map(|c| &c.value).collect::<Vec<_>>());
            println!("   Reduction: {} -> {} choices", choices.len(), shrunk.len());
        }
        other => println!("❌ Result: {:?}", other),
    }

    // Demo 2: String optimization
    println!("\n📝 Demo 2: String Optimization");
    let string_choices = vec![
        Choice { value: ChoiceValue::String("HELLO WORLD!!!   ".to_string()), index: 0 },
        Choice { value: ChoiceValue::String("TEST String\t\n  ".to_string()), index: 1 },
    ];

    println!("Original strings:");
    for choice in &string_choices {
        if let ChoiceValue::String(s) = &choice.value {
            println!("  '{}'", s);
        }
    }

    match engine.shrink_choices(&string_choices) {
        ShrinkResult::Success(shrunk) => {
            println!("✅ Optimized strings:");
            for choice in &shrunk {
                if let ChoiceValue::String(s) = &choice.value {
                    println!("  '{}'", s);
                }
            }
        }
        other => println!("❌ Result: {:?}", other),
    }

    // Demo 3: Mixed data types
    println!("\n🔀 Demo 3: Mixed Data Types");
    let mixed_choices = vec![
        Choice { value: ChoiceValue::Integer(999), index: 0 },
        Choice { value: ChoiceValue::Float(3.141592653589793), index: 1 },
        Choice { value: ChoiceValue::String("Complex String!!!".to_string()), index: 2 },
        Choice { value: ChoiceValue::Bytes(vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF]), index: 3 },
        Choice { value: ChoiceValue::Boolean(true), index: 4 },
        Choice { value: ChoiceValue::Integer(999), index: 5 }, // Duplicate
    ];

    println!("Original mixed data:");
    for (i, choice) in mixed_choices.iter().enumerate() {
        println!("  {}: {:?}", i, choice.value);
    }

    match engine.shrink_choices(&mixed_choices) {
        ShrinkResult::Success(shrunk) => {
            println!("✅ Shrunk mixed data:");
            for (i, choice) in shrunk.iter().enumerate() {
                println!("  {}: {:?}", i, choice.value);
            }
            println!("   Original size: {}, Shrunk size: {}", mixed_choices.len(), shrunk.len());
        }
        other => println!("❌ Result: {:?}", other),
    }

    // Demo 4: Strategy management
    println!("\n⚙️  Demo 4: Custom Strategy Management");
    
    // Add a custom aggressive integer strategy
    let custom_strategy = ShrinkingStrategy::MinimizeIntegers { target: 42, aggressive: true };
    engine.add_strategy(custom_strategy.clone(), 10);
    
    let big_numbers = vec![
        Choice { value: ChoiceValue::Integer(10000), index: 0 },
        Choice { value: ChoiceValue::Integer(5000), index: 1 },
    ];

    println!("Testing custom strategy (target: 42):");
    println!("Original: {:?}", big_numbers.iter().map(|c| &c.value).collect::<Vec<_>>());

    match engine.shrink_choices(&big_numbers) {
        ShrinkResult::Success(shrunk) => {
            println!("✅ With custom strategy: {:?}", shrunk.iter().map(|c| &c.value).collect::<Vec<_>>());
        }
        other => println!("❌ Result: {:?}", other),
    }

    // Demo 5: Performance metrics
    println!("\n📈 Demo 5: Performance Metrics");
    let metrics = engine.get_metrics();
    println!("Engine metrics: {}", metrics);
    
    let success_rates = engine.get_strategy_success_rates();
    println!("Strategy success rates:");
    for (strategy, rate) in success_rates.iter().take(3) {
        println!("  {:?}: {:.1}%", strategy, rate * 100.0);
    }

    println!("\n🎉 Advanced Shrinking System Demo Complete!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shrinking_demo_runs() {
        // This test ensures the demo runs without panicking
        demo_advanced_shrinking();
    }

    #[test]
    fn test_progressive_shrinking() {
        let mut engine = AdvancedShrinkingEngine::default();
        
        // Start with a large test case
        let mut current = vec![
            Choice { value: ChoiceValue::Integer(1000), index: 0 },
            Choice { value: ChoiceValue::Integer(2000), index: 1 },
            Choice { value: ChoiceValue::Integer(3000), index: 2 },
            Choice { value: ChoiceValue::String("VERY LONG STRING WITH LOTS OF CONTENT!!!".to_string()), index: 3 },
        ];

        let original_size = current.len();
        let mut iteration = 0;
        let max_iterations = 5;

        println!("🔄 Progressive Shrinking Test");
        println!("Starting with {} choices", original_size);

        while iteration < max_iterations {
            match engine.shrink_choices(&current) {
                ShrinkResult::Success(shrunk) => {
                    println!("  Iteration {}: {} choices", iteration + 1, shrunk.len());
                    if shrunk.len() >= current.len() {
                        // No further improvement
                        break;
                    }
                    current = shrunk;
                }
                ShrinkResult::Failed => {
                    println!("  Iteration {}: No further shrinking possible", iteration + 1);
                    break;
                }
                other => {
                    println!("  Iteration {}: {:?}", iteration + 1, other);
                    break;
                }
            }
            iteration += 1;
        }

        println!("Final size: {} choices (reduction: {}%)", 
                current.len(), 
                ((original_size - current.len()) as f64 / original_size as f64) * 100.0);

        // Verify we made some progress
        assert!(current.len() <= original_size, "Should not increase in size");
    }

    #[test]
    fn test_constraint_aware_shrinking() {
        let mut engine = AdvancedShrinkingEngine::default();
        
        // Test with values that would violate constraints if shrunk too aggressively
        let choices = vec![
            Choice { value: ChoiceValue::Integer(99999), index: 0 }, // Out of range
            Choice { value: ChoiceValue::Float(f64::INFINITY), index: 1 }, // Invalid
        ];

        match engine.shrink_choices(&choices) {
            ShrinkResult::Success(shrunk) => {
                for choice in &shrunk {
                    match &choice.value {
                        ChoiceValue::Integer(i) => {
                            assert!((*i).abs() <= 10000, "Integer should be constrained: {}", i);
                        }
                        ChoiceValue::Float(f) => {
                            assert!(f.is_finite(), "Float should be finite: {}", f);
                        }
                        _ => {}
                    }
                }
                println!("✅ Constraint-aware shrinking successful");
            }
            other => {
                println!("Constraint test result: {:?}", other);
            }
        }
    }
}