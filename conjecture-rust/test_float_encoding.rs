use conjecture_rust::choice::shrinking_system::AdvancedShrinkingEngine;
use conjecture_rust::float_encoding_export::{float_to_lex, lex_to_float, float_to_int, int_to_float};

fn main() {
    println!("Testing Float Encoding Export Integration");
    
    // Test basic float encoding
    let test_values = [0.0, 1.0, 3.14159, -2.718281828, f64::MAX, f64::MIN_POSITIVE];
    
    for value in &test_values {
        // Test lexicographic encoding
        let lex = float_to_lex(*value);
        let recovered_lex = lex_to_float(lex);
        
        // Test integer storage
        let int_repr = float_to_int(*value);
        let recovered_int = int_to_float(int_repr);
        
        println!("Value: {:.10}, Lex: 0x{:016X} -> {:.10}, Int: 0x{:016X} -> {:.10}", 
                 value, lex, recovered_lex, int_repr, recovered_int);
        
        if value.is_finite() {
            assert_eq!(*value, recovered_lex, "Lex encoding should roundtrip");
            assert_eq!(*value, recovered_int, "Int conversion should roundtrip");
        }
    }
    
    // Test ShrinkingEngine float encoding integration
    let mut engine = AdvancedShrinkingEngine::default();
    
    // Test DataTree storage integration
    let test_float = 2.718281828;
    let stored = engine.float_to_datatree_storage(test_float);
    let restored = engine.float_from_datatree_storage(stored);
    assert_eq!(test_float, restored);
    println!("DataTree storage test: {} -> 0x{:016X} -> {}", test_float, stored, restored);
    
    // Test lexicographic encoding
    let lex_encoding = engine.get_float_lex_encoding(test_float);
    let decoded = engine.decode_float_lex_encoding(lex_encoding);
    assert_eq!(test_float, decoded);
    println!("Lex encoding test: {} -> 0x{:016X} -> {}", test_float, lex_encoding, decoded);
    
    println!("✓ All float encoding integration tests passed!");
}