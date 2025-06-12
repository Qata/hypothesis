// Test script to verify FloatConstraints Option<f64> type system fix
use conjecture_rust::choice::{FloatConstraints, Constraints};

fn main() {
    println!("Testing FloatConstraints Option<f64> type system fix...");

    // Test 1: Default constructor should use Option<f64>
    let default_constraints = FloatConstraints::default();
    println!("✓ Default constraints created: smallest_nonzero_magnitude = {:?}", 
             default_constraints.smallest_nonzero_magnitude);

    // Test 2: Create with explicit None
    let constraints_none = FloatConstraints {
        min_value: -1.0,
        max_value: 1.0,
        allow_nan: false,
        smallest_nonzero_magnitude: None,
    };
    println!("✓ None constraints: smallest_nonzero_magnitude = {:?}", 
             constraints_none.smallest_nonzero_magnitude);

    // Test 3: Create with explicit Some(value)
    let constraints_some = FloatConstraints {
        min_value: -1.0,
        max_value: 1.0,
        allow_nan: false,
        smallest_nonzero_magnitude: Some(1e-6),
    };
    println!("✓ Some constraints: smallest_nonzero_magnitude = {:?}", 
             constraints_some.smallest_nonzero_magnitude);

    // Test 4: Constructor function with Option<f64>
    match FloatConstraints::with_smallest_nonzero_magnitude(
        Some(-2.0),
        Some(2.0),
        false,
        Some(1e-5), // Option<f64> parameter
    ) {
        Ok(constraints) => {
            println!("✓ Constructor with Some: smallest_nonzero_magnitude = {:?}", 
                     constraints.smallest_nonzero_magnitude);
        }
        Err(e) => {
            println!("✗ Constructor with Some failed: {}", e);
        }
    }

    // Test 5: Constructor function with None
    match FloatConstraints::with_smallest_nonzero_magnitude(
        Some(-2.0),
        Some(2.0),
        false,
        None, // None parameter
    ) {
        Ok(constraints) => {
            println!("✓ Constructor with None: smallest_nonzero_magnitude = {:?}", 
                     constraints.smallest_nonzero_magnitude);
        }
        Err(e) => {
            println!("✗ Constructor with None failed: {}", e);
        }
    }

    // Test 6: Clone preserves Option<f64>
    let cloned = constraints_some.clone();
    println!("✓ Cloned constraints: smallest_nonzero_magnitude = {:?}", 
             cloned.smallest_nonzero_magnitude);

    // Test 7: Equality works with Option<f64>
    let same_constraints = FloatConstraints {
        min_value: -1.0,
        max_value: 1.0,
        allow_nan: false,
        smallest_nonzero_magnitude: Some(1e-6),
    };
    
    if constraints_some == same_constraints {
        println!("✓ Equality works with Option<f64>");
    } else {
        println!("✗ Equality failed with Option<f64>");
    }

    // Test 8: Integration with Constraints enum
    let choice_constraints = Constraints::Float(constraints_some);
    if let Constraints::Float(fc) = &choice_constraints {
        println!("✓ Integration with Constraints enum: smallest_nonzero_magnitude = {:?}", 
                 fc.smallest_nonzero_magnitude);
    }

    println!("\nFloatConstraints Option<f64> type system verification completed!");
}