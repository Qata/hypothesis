// Simple float encoding verification without PyO3 dependencies
// Tests the Rust float encoding functions for basic correctness

use conjecture::choice::indexing::float_encoding::{float_to_lex, lex_to_float};

/// Test basic float encoding functionality
pub fn run_simple_float_encoding_verification() -> Result<(), String> {
    println!("🔢 Running simple float encoding verification...");
    
    let test_values = vec![
        0.0, -0.0, 1.0, -1.0, 2.5, -2.5,
        1e-100, 1e100, 1.2345e-50, -9.8765e50,
        2.2250738585072014e-308,  // min normal f64
        1.7976931348623157e308,   // max f64
        f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
    ];
    
    let mut passed = 0;
    let mut failed = 0;
    
    // Test round-trip encoding/decoding
    println!("Testing round-trip consistency:");
    for &value in &test_values {
        let encoded = float_to_lex(value);
        let decoded = lex_to_float(encoded);
        
        let matches = if value.is_nan() && decoded.is_nan() {
            true
        } else if value.is_infinite() && decoded.is_infinite() {
            value.signum() == decoded.signum()
        } else {
            (value - decoded).abs() < f64::EPSILON || (value == 0.0 && decoded == 0.0)
        };
        
        if matches {
            passed += 1;
            println!("  ✓ {} -> {:016X} -> {}", value, encoded, decoded);
        } else {
            failed += 1;
            println!("  ✗ {} -> {:016X} -> {} (mismatch)", value, encoded, decoded);
        }
    }
    
    // Test lexicographic ordering properties
    println!("\nTesting lexicographic ordering:");
    let ordering_tests = vec![
        (0.0, 1.0),
        (-1.0, 0.0),  
        (1.0, 2.0),
        (-2.0, -1.0),
        (1.0, f64::INFINITY),
        (f64::NEG_INFINITY, 0.0),
    ];
    
    for &(a, b) in &ordering_tests {
        let lex_a = float_to_lex(a);
        let lex_b = float_to_lex(b);
        
        // For basic ordering tests, lexicographic order should match numeric order
        let lex_ordered = lex_a < lex_b;
        let num_ordered = a < b;
        
        if lex_ordered == num_ordered {
            passed += 1;
            println!("  ✓ {} < {} : {:016X} < {:016X}", a, b, lex_a, lex_b);
        } else {
            failed += 1;
            println!("  ✗ {} < {} : {:016X} < {:016X} (ordering mismatch)", a, b, lex_a, lex_b);
        }
    }
    
    // Test special value encodings
    println!("\nTesting special value encodings:");
    let special_tests = vec![
        (0.0, "Zero"),
        (f64::NAN, "NaN"),
        (f64::INFINITY, "Positive infinity"),
        (f64::NEG_INFINITY, "Negative infinity"),
    ];
    
    for &(value, description) in &special_tests {
        let encoded = float_to_lex(value);
        let decoded = lex_to_float(encoded);
        
        let correct = match description {
            "Zero" => decoded == 0.0,
            "NaN" => decoded.is_nan(),
            "Positive infinity" => decoded.is_infinite() && decoded.is_sign_positive(),
            "Negative infinity" => decoded.is_infinite() && decoded.is_sign_negative(),
            _ => false,
        };
        
        if correct {
            passed += 1;
            println!("  ✓ {} ({}): {:016X}", value, description, encoded);
        } else {
            failed += 1;
            println!("  ✗ {} ({}): {:016X} -> {} (incorrect)", value, description, encoded, decoded);
        }
    }
    
    println!("\n=== SIMPLE VERIFICATION RESULTS ===");
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);
    
    if failed == 0 {
        println!("✅ All simple float encoding tests passed!");
        Ok(())
    } else {
        Err(format!("❌ {} test(s) failed", failed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_verification() {
        assert!(run_simple_float_encoding_verification().is_ok());
    }
}