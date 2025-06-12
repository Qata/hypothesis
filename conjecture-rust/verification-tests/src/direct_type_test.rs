//! Direct unit test for Float Constraint Type System Consistency
//! This test runs without Python dependencies to verify Rust type consistency

use conjecture::choice::{FloatConstraints, ChoiceValue, Constraints};

/// Run Float Constraint Type System Consistency Tests without Python
pub fn run_direct_type_tests() -> Result<(), String> {
    println!("🚀 Running Float Constraint Type System Consistency Tests (Direct Rust)...");
    
    // Test 1: Direct field access (f64 vs Option<f64>)
    println!("  Test 1: Direct field access type consistency");
    let constraints = FloatConstraints::default();
    
    // This should handle Option<f64> properly
    let magnitude: Option<f64> = constraints.smallest_nonzero_magnitude;
    if magnitude.map_or(false, |m| m <= 0.0) {
        return Err("Default smallest_nonzero_magnitude should be positive".to_string());
    }
    println!("    ✓ Default smallest_nonzero_magnitude: {:?}", magnitude);
    
    // Test 2: Assignment should accept f64 directly
    println!("  Test 2: Direct f64 assignment");
    let custom_constraints = FloatConstraints {
        min_value: -10.0,
        max_value: 10.0,
        allow_nan: false,
        smallest_nonzero_magnitude: Some(1e-6), // Direct f64 assignment
    };
    
    if custom_constraints.smallest_nonzero_magnitude != Some(1e-6) {
        return Err("Direct f64 assignment failed".to_string());
    }
    println!("    ✓ Custom smallest_nonzero_magnitude: {:?}", custom_constraints.smallest_nonzero_magnitude);
    
    // Test 3: Constructor should accept f64 directly  
    println!("  Test 3: Constructor with f64 parameter");
    let constructed = FloatConstraints::with_smallest_nonzero_magnitude(
        Some(-100.0),
        Some(100.0),
        true,
        Some(1e-8), // Option<f64> value
    ).map_err(|e| format!("Constructor failed: {}", e))?;
    
    if constructed.smallest_nonzero_magnitude != Some(1e-8) {
        return Err("Constructor f64 parameter failed".to_string());
    }
    println!("    ✓ Constructed smallest_nonzero_magnitude: {:?}", constructed.smallest_nonzero_magnitude);
    
    // Test 4: Cloning preserves type
    println!("  Test 4: Cloning preserves f64 type");
    let cloned = constructed.clone();
    let cloned_magnitude: Option<f64> = cloned.smallest_nonzero_magnitude;
    if cloned_magnitude != Some(1e-8) {
        return Err("Cloning did not preserve Option<f64> value".to_string());
    }
    println!("    ✓ Cloned smallest_nonzero_magnitude: {:?}", cloned_magnitude);
    
    // Test 5: Validation with magnitude constraint
    println!("  Test 5: Validation logic with magnitude constraint");
    let validation_constraints = FloatConstraints {
        min_value: -10.0,
        max_value: 10.0,
        allow_nan: false,
        smallest_nonzero_magnitude: Some(1e-3),
    };
    
    // Test valid values
    let valid_tests = vec![
        (0.0, "Zero should always be valid"),
        (1e-3, "Exactly at boundary should be valid"),
        (-1e-3, "Negative boundary should be valid"),
        (1e-2, "Above boundary should be valid"),
        (5.0, "Normal value should be valid"),
    ];
    
    for (value, desc) in valid_tests {
        if !validation_constraints.validate(value) {
            return Err(format!("Validation failed for valid value {}: {}", value, desc));
        }
    }
    
    // Test invalid values
    let invalid_tests = vec![
        (1e-4, "Below magnitude threshold should be invalid"),
        (-1e-4, "Below negative magnitude threshold should be invalid"),
        (f64::NAN, "NaN should be invalid when not allowed"),
        (15.0, "Above max should be invalid"),
        (-15.0, "Below min should be invalid"),
    ];
    
    for (value, desc) in invalid_tests {
        if value.is_nan() {
            // NaN comparison is special
            if validation_constraints.validate(value) {
                return Err(format!("Validation incorrectly passed for NaN: {}", desc));
            }
        } else if validation_constraints.validate(value) {
            return Err(format!("Validation incorrectly passed for invalid value {}: {}", value, desc));
        }
    }
    
    println!("    ✓ Validation logic works correctly with f64 magnitude");
    
    // Test 6: Enum wrapper compatibility
    println!("  Test 6: Enum wrapper preserves f64 type");
    let float_constraints = FloatConstraints {
        min_value: 0.0,
        max_value: 1.0,
        allow_nan: false,
        smallest_nonzero_magnitude: Some(1e-6),
    };
    
    let constraints_enum = Constraints::Float(float_constraints.clone());
    
    // Verify the enum contains the correct constraints
    if let Constraints::Float(ref c) = constraints_enum {
        let magnitude: Option<f64> = c.smallest_nonzero_magnitude; // Direct Option<f64> access
        if magnitude != Some(1e-6) {
            return Err("Enum wrapper did not preserve f64 value".to_string());
        }
        println!("    ✓ Enum wrapper preserves Option<f64> type: {:?}", magnitude);
    } else {
        return Err("Expected Float constraints in enum".to_string());
    }
    
    // Test 7: Edge case values
    println!("  Test 7: Edge case magnitude values");
    let edge_cases = vec![
        f64::MIN_POSITIVE,           // Smallest normal
        2.2250738585072014e-308,     // Python's SMALLEST_SUBNORMAL
        1e-100,                      // Very small
        1e-6,                        // Common test value
        1.0,                         // Large value
    ];
    
    for magnitude in edge_cases {
        let edge_constraints = FloatConstraints {
            min_value: f64::NEG_INFINITY,
            max_value: f64::INFINITY,
            allow_nan: true,
            smallest_nonzero_magnitude: Some(magnitude), // Option<f64> assignment
        };
        
        // Verify direct access works
        let _direct_access: Option<f64> = edge_constraints.smallest_nonzero_magnitude;
        if edge_constraints.smallest_nonzero_magnitude != Some(magnitude) {
            return Err(format!("Edge case magnitude {} not preserved", magnitude));
        }
    }
    println!("    ✓ All edge case magnitude values handled correctly");
    
    println!("✅ Float Constraint Type System Consistency Tests: ALL PASSED!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_constraint_type_consistency_direct() {
        match run_direct_type_tests() {
            Ok(()) => {
                println!("🎉 Direct Float Constraint Type System tests passed!");
            }
            Err(e) => {
                panic!("Direct Float Constraint Type System tests failed: {}", e);
            }
        }
    }
}