// Test script to verify that our sophisticated float generation works
use conjecture::{ConjectureData, draw_float_for_provider, FloatWidth};
use conjecture::choice::FloatConstraints;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    println!("Testing sophisticated float generation integration...");
    
    // Test 1: Direct advanced function call
    println!("\n=== Test 1: Direct Advanced Float Generation ===");
    let mut rng = ChaCha8Rng::from_seed([42u8; 32]);
    let constraints = FloatConstraints {
        min_value: 0.0,
        max_value: 1.0,
        allow_nan: false,
        smallest_nonzero_magnitude: None,
    };
    
    for i in 0..10 {
        match draw_float_for_provider(&mut rng, &constraints) {
            Ok(value) => println!("  Generated float {}: {}", i, value),
            Err(e) => println!("  Error {}: {}", i, e),
        }
    }
    
    // Test 2: Provider-based generation (through ConjectureData)
    println!("\n=== Test 2: Provider-Based Float Generation ===");
    let mut data = ConjectureData::new(123);
    
    for i in 0..10 {
        match data.draw_float(0.0, 10.0, false, None, None, true) {
            Ok(value) => println!("  Provider generated float {}: {}", i, value),
            Err(e) => println!("  Provider error {}: {:?}", i, e),
        }
    }
    
    // Test 3: Verify lexicographic properties
    println!("\n=== Test 3: Lexicographic Properties ===");
    use conjecture::{float_to_lex, lex_to_float};
    let test_values = [0.0, 1.0, 2.0, 0.5, -1.0, f64::consts::PI];
    
    for &val in &test_values {
        let lex = float_to_lex(val, FloatWidth::Width64);
        let recovered = lex_to_float(lex, FloatWidth::Width64);
        println!("  {} -> lex: {} -> recovered: {}", val, lex, recovered);
    }
    
    println!("\n✅ Float generation integration test completed!");
}