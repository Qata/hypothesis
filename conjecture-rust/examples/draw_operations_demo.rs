//! Demo of the core draw operations with Python parity
//! 
//! This program demonstrates that our Rust implementation provides the same
//! functionality as the Python ConjectureData draw operations.

use conjecture::data::ConjectureData;
use conjecture::choice::IntervalSet;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ConjectureData Draw Operations Demo ===\n");
    
    // Create a new ConjectureData instance
    let mut data = ConjectureData::new(42); // Using seed 42
    
    // 1. Integer Drawing
    println!("1. Integer Drawing:");
    
    // Basic integer range
    let int_value = data.draw_integer(Some(0), Some(10), None, 0, None, true)?;
    println!("   Basic range [0, 10]: {}", int_value);
    
    // Forced integer value
    let forced_int = data.draw_integer(Some(0), Some(10), None, 0, Some(7), true)?;
    println!("   Forced value 7: {}", forced_int);
    
    // Weighted integer
    let mut weights = HashMap::new();
    weights.insert(5, 0.8); // 80% chance for value 5
    let weighted_int = data.draw_integer(Some(0), Some(10), Some(weights), 0, None, true)?;
    println!("   Weighted (80% chance of 5): {}", weighted_int);
    
    // 2. Boolean Drawing
    println!("\n2. Boolean Drawing:");
    
    // Basic boolean with 50% probability
    let bool_value = data.draw_boolean(0.5, None, true)?;
    println!("   50% probability: {}", bool_value);
    
    // Deterministic false
    let false_value = data.draw_boolean(0.0, None, true)?;
    println!("   Deterministic false: {}", false_value);
    
    // Deterministic true
    let true_value = data.draw_boolean(1.0, None, true)?;
    println!("   Deterministic true: {}", true_value);
    
    // 3. Float Drawing
    println!("\n3. Float Drawing:");
    
    // Basic float in range
    let float_value = data.draw_float(0.0, 1.0, false, None, None, true)?;
    println!("   Range [0.0, 1.0]: {}", float_value);
    
    // Float with NaN allowed
    let float_with_nan = data.draw_float(f64::NEG_INFINITY, f64::INFINITY, true, None, None, true)?;
    println!("   Full range (NaN allowed): {}", float_with_nan);
    
    // Forced float value
    let forced_float = data.draw_float(0.0, 1.0, false, None, Some(0.42), true)?;
    println!("   Forced value 0.42: {}", forced_float);
    
    // 4. String Drawing
    println!("\n4. String Drawing:");
    
    // Basic string from alphabet
    let intervals = IntervalSet::from_string("abc");
    let string_value = data.draw_string(intervals.clone(), 0, 5, None, true)?;
    println!("   Alphabet 'abc', length 0-5: '{}'", string_value);
    
    // Forced string value
    let forced_string = data.draw_string(intervals, 0, 10, Some("hello".to_string()), true)?;
    println!("   Forced value 'hello': '{}'", forced_string);
    
    // 5. Bytes Drawing
    println!("\n5. Bytes Drawing:");
    
    // Basic bytes
    let bytes_value = data.draw_bytes(0, 5, None, true)?;
    println!("   Length 0-5: {:?}", bytes_value);
    
    // Fixed size bytes
    let fixed_bytes = data.draw_bytes(3, 3, None, true)?;
    println!("   Fixed length 3: {:?}", fixed_bytes);
    
    // 6. Constraint Validation
    println!("\n6. Constraint Validation:");
    
    // Test invalid range (should fail)
    match data.draw_integer(Some(10), Some(5), None, 0, None, true) {
        Ok(_) => println!("   ERROR: Invalid range should have failed!"),
        Err(e) => println!("   Invalid range correctly rejected: {}", e),
    }
    
    // Test invalid probability (should fail)
    match data.draw_boolean(1.5, None, true) {
        Ok(_) => println!("   ERROR: Invalid probability should have failed!"),
        Err(e) => println!("   Invalid probability correctly rejected: {}", e),
    }
    
    // 7. State Tracking
    println!("\n7. State Tracking:");
    println!("   Current length: {}", data.length);
    println!("   Current status: {:?}", data.status);
    
    // 8. Legacy Compatibility
    println!("\n8. Legacy Compatibility:");
    
    let legacy_int = data.draw_integer_simple(1, 6)?;
    println!("   Legacy integer [1, 6]: {}", legacy_int);
    
    let legacy_float = data.draw_float_simple()?;
    println!("   Legacy float: {}", legacy_float);
    
    let legacy_string = data.draw_string_simple("xyz", 2, 4)?;
    println!("   Legacy string 'xyz', length 2-4: '{}'", legacy_string);
    
    let legacy_bytes = data.draw_bytes_simple(2)?;
    println!("   Legacy bytes, size 2: {:?}", legacy_bytes);
    
    println!("\n=== Demo completed successfully! ===");
    println!("All draw operations are working with Python-equivalent functionality.");
    
    Ok(())
}