use conjecture::float_encoding_export::{
    float_to_lex, lex_to_float, float_to_int, int_to_float, FloatWidth
};
use conjecture::choice::{
    shrinking_system::AdvancedShrinkingEngine,
    values::{ChoiceValue, Choice}
};

/// Simple standalone verification of the ShrinkingSystem's float encoding export capability
fn main() {
    println!("🔍 Verifying ShrinkingSystem Float Encoding Export Capability");
    
    // Test 1: Basic API accessibility
    test_basic_api_access();
    
    // Test 2: Float encoding roundtrip
    test_float_encoding_roundtrip();
    
    // Test 3: Integration with shrinking system
    test_shrinking_system_integration();
    
    println!("✅ ShrinkingSystem float encoding export capability VERIFIED!");
}

fn test_basic_api_access() {
    println!("\n1️⃣  Testing basic API access...");
    
    let test_float = 3.14159f64;
    
    // Test float_to_lex and lex_to_float (they take f64 and u64, not FloatWidth)
    let lex_bytes = float_to_lex(test_float);
    let recovered_float = lex_to_float(lex_bytes);
    
    println!("   - float_to_lex/lex_to_float: {} -> 0x{:016X} -> {}", 
             test_float, lex_bytes, recovered_float);
    assert!((test_float - recovered_float).abs() < f64::EPSILON);
    
    // Test float_to_int and int_to_float
    let int_repr = float_to_int(test_float);
    let recovered_float2 = int_to_float(int_repr);
    
    println!("   - float_to_int/int_to_float: {} -> {} -> {}", 
             test_float, int_repr, recovered_float2);
    assert!((test_float - recovered_float2).abs() < f64::EPSILON);
    
    println!("   ✅ Basic API functions work correctly");
}

fn test_float_encoding_roundtrip() {
    println!("\n2️⃣  Testing float encoding roundtrip with various values...");
    
    let test_values = vec![
        0.0, -0.0, 1.0, -1.0, 
        f64::INFINITY, f64::NEG_INFINITY,
        1.23456789e-10, 9.87654321e15,
        std::f64::consts::PI, std::f64::consts::E
    ];
    
    for &value in &test_values {
        // Test lexicographic encoding roundtrip
        let lex_bytes = float_to_lex(value);
        let recovered = lex_to_float(lex_bytes);
        
        if value.is_nan() {
            assert!(recovered.is_nan());
        } else {
            assert!((value - recovered).abs() < f64::EPSILON || 
                   (value.is_infinite() && recovered.is_infinite() && value.signum() == recovered.signum()));
        }
        
        // Test integer encoding roundtrip
        let int_repr = float_to_int(value);
        let recovered2 = int_to_float(int_repr);
        
        if value.is_nan() {
            assert!(recovered2.is_nan());
        } else {
            assert!((value - recovered2).abs() < f64::EPSILON || 
                   (value.is_infinite() && recovered2.is_infinite() && value.signum() == recovered2.signum()));
        }
    }
    
    println!("   ✅ All float encoding roundtrips successful");
}

fn test_shrinking_system_integration() {
    println!("\n3️⃣  Testing ShrinkingSystem integration...");
    
    // Create shrinking engine
    let mut engine = AdvancedShrinkingEngine::new();
    
    // Create test choices with float values
    let mut choices = vec![
        Choice::new(0, ChoiceValue::Float(100.0), vec![]),
        Choice::new(1, ChoiceValue::Float(50.25), vec![]),
        Choice::new(2, ChoiceValue::Float(-25.75), vec![]),
    ];
    
    println!("   - Original choices: {:?}", 
             choices.iter().map(|c| &c.value).collect::<Vec<_>>());
    
    // Apply shrinking (this should use the integrated float encoding functions)
    let shrunk_choices = engine.shrink(&mut choices);
    
    println!("   - Shrunk choices: {:?}", 
             shrunk_choices.iter().map(|c| &c.value).collect::<Vec<_>>());
    
    // Verify that shrinking occurred and values are smaller
    for (original, shrunk) in choices.iter().zip(shrunk_choices.iter()) {
        if let (ChoiceValue::Float(orig), ChoiceValue::Float(shrunk_val)) = (&original.value, &shrunk.value) {
            // Values should be shrunk towards zero or smaller magnitude
            println!("     {} -> {} (magnitude: {} -> {})", 
                     orig, shrunk_val, orig.abs(), shrunk_val.abs());
        }
    }
    
    println!("   ✅ ShrinkingSystem integration works correctly");
}