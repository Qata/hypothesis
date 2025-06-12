//! Test case definitions for verification

use conjecture::choice::*;

/// A single verification test case
#[derive(Debug, Clone)]
pub struct TestCase {
    pub name: String,
    pub description: String,
    pub value: ChoiceValue,
    pub constraints: Constraints,
    pub expected_properties: Vec<String>,
}

/// Test suite containing multiple test cases
#[derive(Debug, Clone)]
pub struct TestSuite {
    pub name: String,
    pub cases: Vec<TestCase>,
}

/// Generate comprehensive test suites
pub fn get_all_test_suites() -> Vec<TestSuite> {
    vec![
        get_integer_test_suite(),
        get_boolean_test_suite(),
        get_float_test_suite(),
        get_edge_case_test_suite(),
    ]
}

/// Integer choice indexing test cases
pub fn get_integer_test_suite() -> TestSuite {
    let cases = vec![
        // Unbounded cases with default shrink_towards=0
        TestCase {
            name: "integer_unbounded_zero".to_string(),
            description: "Unbounded integer with value 0 should have index 0".to_string(),
            value: ChoiceValue::Integer(0),
            constraints: Constraints::Integer(IntegerConstraints {
                min_value: None,
                max_value: None,
                weights: None,
                shrink_towards: Some(0),
            }),
            expected_properties: vec!["index_zero".to_string(), "roundtrip".to_string()],
        },
        TestCase {
            name: "integer_unbounded_positive".to_string(),
            description: "Unbounded integer positive values follow zigzag pattern".to_string(),
            value: ChoiceValue::Integer(1),
            constraints: Constraints::Integer(IntegerConstraints {
                min_value: None,
                max_value: None,
                weights: None,
                shrink_towards: Some(0),
            }),
            expected_properties: vec!["index_one".to_string(), "roundtrip".to_string()],
        },
        TestCase {
            name: "integer_unbounded_negative".to_string(),
            description: "Unbounded integer negative values follow zigzag pattern".to_string(),
            value: ChoiceValue::Integer(-1),
            constraints: Constraints::Integer(IntegerConstraints {
                min_value: None,
                max_value: None,
                weights: None,
                shrink_towards: Some(0),
            }),
            expected_properties: vec!["index_two".to_string(), "roundtrip".to_string()],
        },
        
        // Custom shrink_towards
        TestCase {
            name: "integer_custom_shrink_towards".to_string(),
            description: "Custom shrink_towards value should have index 0".to_string(),
            value: ChoiceValue::Integer(5),
            constraints: Constraints::Integer(IntegerConstraints {
                min_value: None,
                max_value: None,
                weights: None,
                shrink_towards: Some(5),
            }),
            expected_properties: vec!["index_zero".to_string(), "roundtrip".to_string()],
        },
        
        // Bounded cases
        TestCase {
            name: "integer_bounded_shrink_towards_in_range".to_string(),
            description: "Bounded integer with shrink_towards in range".to_string(),
            value: ChoiceValue::Integer(0),
            constraints: Constraints::Integer(IntegerConstraints {
                min_value: Some(-3),
                max_value: Some(3),
                weights: None,
                shrink_towards: Some(0),
            }),
            expected_properties: vec!["index_zero".to_string(), "roundtrip".to_string()],
        },
        
        // Shrink_towards clamping
        TestCase {
            name: "integer_shrink_towards_clamped_to_min".to_string(),
            description: "shrink_towards below min_value should be clamped".to_string(),
            value: ChoiceValue::Integer(1),
            constraints: Constraints::Integer(IntegerConstraints {
                min_value: Some(1),
                max_value: Some(5),
                weights: None,
                shrink_towards: Some(0), // Will be clamped to 1
            }),
            expected_properties: vec!["index_zero".to_string(), "roundtrip".to_string()],
        },
    ];
    
    TestSuite {
        name: "Integer Choice Indexing".to_string(),
        cases,
    }
}

/// Boolean choice indexing test cases
pub fn get_boolean_test_suite() -> TestSuite {
    let cases = vec![
        TestCase {
            name: "boolean_p_zero_false_only".to_string(),
            description: "p=0.0 should only permit false".to_string(),
            value: ChoiceValue::Boolean(false),
            constraints: Constraints::Boolean(BooleanConstraints { p: 0.0 }),
            expected_properties: vec!["index_zero".to_string(), "roundtrip".to_string()],
        },
        TestCase {
            name: "boolean_p_one_true_only".to_string(),
            description: "p=1.0 should only permit true".to_string(),
            value: ChoiceValue::Boolean(true),
            constraints: Constraints::Boolean(BooleanConstraints { p: 1.0 }),
            expected_properties: vec!["index_zero".to_string(), "roundtrip".to_string()],
        },
        TestCase {
            name: "boolean_p_half_false".to_string(),
            description: "p=0.5 false should have index 0".to_string(),
            value: ChoiceValue::Boolean(false),
            constraints: Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            expected_properties: vec!["index_zero".to_string(), "roundtrip".to_string()],
        },
        TestCase {
            name: "boolean_p_half_true".to_string(),
            description: "p=0.5 true should have index 1".to_string(),
            value: ChoiceValue::Boolean(true),
            constraints: Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            expected_properties: vec!["index_one".to_string(), "roundtrip".to_string()],
        },
    ];
    
    TestSuite {
        name: "Boolean Choice Indexing".to_string(),
        cases,
    }
}

/// Float choice indexing test cases
pub fn get_float_test_suite() -> TestSuite {
    let cases = vec![
        TestCase {
            name: "float_positive_zero".to_string(),
            description: "Positive zero should have sign bit 0".to_string(),
            value: ChoiceValue::Float(0.0),
            constraints: Constraints::Float(FloatConstraints {
                min_value: f64::NEG_INFINITY,
                max_value: f64::INFINITY,
                allow_nan: true,
                smallest_nonzero_magnitude: f64::MIN_POSITIVE,
            }),
            expected_properties: vec!["sign_bit_zero".to_string(), "roundtrip".to_string()],
        },
        TestCase {
            name: "float_negative_zero".to_string(),
            description: "Negative zero should have sign bit 1".to_string(),
            value: ChoiceValue::Float(-0.0),
            constraints: Constraints::Float(FloatConstraints {
                min_value: f64::NEG_INFINITY,
                max_value: f64::INFINITY,
                allow_nan: true,
                smallest_nonzero_magnitude: f64::MIN_POSITIVE,
            }),
            expected_properties: vec!["sign_bit_one".to_string(), "roundtrip".to_string()],
        },
        TestCase {
            name: "float_positive_one".to_string(),
            description: "Positive one should have sign bit 0".to_string(),
            value: ChoiceValue::Float(1.0),
            constraints: Constraints::Float(FloatConstraints {
                min_value: f64::NEG_INFINITY,
                max_value: f64::INFINITY,
                allow_nan: true,
                smallest_nonzero_magnitude: f64::MIN_POSITIVE,
            }),
            expected_properties: vec!["sign_bit_zero".to_string(), "roundtrip".to_string()],
        },
        TestCase {
            name: "float_negative_one".to_string(),
            description: "Negative one should have sign bit 1".to_string(),
            value: ChoiceValue::Float(-1.0),
            constraints: Constraints::Float(FloatConstraints {
                min_value: f64::NEG_INFINITY,
                max_value: f64::INFINITY,
                allow_nan: true,
                smallest_nonzero_magnitude: f64::MIN_POSITIVE,
            }),
            expected_properties: vec!["sign_bit_one".to_string(), "roundtrip".to_string()],
        },
    ];
    
    TestSuite {
        name: "Float Choice Indexing".to_string(),
        cases,
    }
}

/// Float Constraint Type System Consistency Tests
pub fn run_float_constraint_type_consistency_tests() {
    println!("🚀 Running Float Constraint Type System Consistency Tests...");
    
    // Test 1: Direct field access (f64 vs Option<f64>)
    println!("  Test 1: Direct field access type consistency");
    let constraints = FloatConstraints::default();
    
    // This should compile without any Option unwrapping
    let magnitude: f64 = constraints.smallest_nonzero_magnitude;
    assert!(magnitude > 0.0);
    println!("    ✓ Default smallest_nonzero_magnitude: {}", magnitude);
    
    // Test 2: Assignment should accept f64 directly
    println!("  Test 2: Direct f64 assignment");
    let custom_constraints = FloatConstraints {
        min_value: -10.0,
        max_value: 10.0,
        allow_nan: false,
        smallest_nonzero_magnitude: 1e-6, // Direct f64 assignment
    };
    
    assert_eq!(custom_constraints.smallest_nonzero_magnitude, 1e-6);
    println!("    ✓ Custom smallest_nonzero_magnitude: {}", custom_constraints.smallest_nonzero_magnitude);
    
    // Test 3: Constructor should accept f64 directly  
    println!("  Test 3: Constructor with f64 parameter");
    let constructed = FloatConstraints::with_smallest_nonzero_magnitude(
        Some(-100.0),
        Some(100.0),
        true,
        1e-8, // Direct f64 value, not Option<f64>
    ).expect("Should create valid constraints");
    
    assert_eq!(constructed.smallest_nonzero_magnitude, 1e-8);
    println!("    ✓ Constructed smallest_nonzero_magnitude: {}", constructed.smallest_nonzero_magnitude);
    
    // Test 4: Cloning preserves type
    println!("  Test 4: Cloning preserves f64 type");
    let cloned = constructed.clone();
    let cloned_magnitude: f64 = cloned.smallest_nonzero_magnitude;
    assert_eq!(cloned_magnitude, 1e-8);
    println!("    ✓ Cloned smallest_nonzero_magnitude: {}", cloned_magnitude);
    
    // Test 5: Validation with magnitude constraint
    println!("  Test 5: Validation logic with magnitude constraint");
    let validation_constraints = FloatConstraints {
        min_value: -10.0,
        max_value: 10.0,
        allow_nan: false,
        smallest_nonzero_magnitude: 1e-3,
    };
    
    // Test valid values
    assert!(validation_constraints.validate(0.0));    // Zero is always valid
    assert!(validation_constraints.validate(1e-3));   // Exactly at boundary
    assert!(validation_constraints.validate(-1e-3));  // Negative boundary
    assert!(validation_constraints.validate(1e-2));   // Above boundary
    assert!(validation_constraints.validate(5.0));    // Normal value
    
    // Test invalid values
    assert!(!validation_constraints.validate(1e-4));  // Below magnitude threshold
    assert!(!validation_constraints.validate(-1e-4)); // Below negative magnitude threshold
    assert!(!validation_constraints.validate(f64::NAN)); // NaN not allowed
    assert!(!validation_constraints.validate(15.0));  // Above max
    assert!(!validation_constraints.validate(-15.0)); // Below min
    
    println!("    ✓ Validation logic works correctly with f64 magnitude");
    
    // Test 6: Enum wrapper compatibility
    println!("  Test 6: Enum wrapper preserves f64 type");
    let float_constraints = FloatConstraints {
        min_value: 0.0,
        max_value: 1.0,
        allow_nan: false,
        smallest_nonzero_magnitude: 1e-6,
    };
    
    let constraints_enum = Constraints::Float(float_constraints.clone());
    
    // Verify the enum contains the correct constraints
    if let Constraints::Float(ref c) = constraints_enum {
        let magnitude: f64 = c.smallest_nonzero_magnitude; // Direct f64 access
        assert_eq!(magnitude, 1e-6);
        println!("    ✓ Enum wrapper preserves f64 type: {}", magnitude);
    } else {
        panic!("Expected Float constraints");
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
            smallest_nonzero_magnitude: magnitude, // Direct f64 assignment
        };
        
        // Verify direct access works
        let _direct_access: f64 = edge_constraints.smallest_nonzero_magnitude;
        assert_eq!(edge_constraints.smallest_nonzero_magnitude, magnitude);
    }
    println!("    ✓ All edge case magnitude values handled correctly");
    
    println!("✅ Float Constraint Type System Consistency Tests: ALL PASSED!");
}

/// Edge case test suite
pub fn get_edge_case_test_suite() -> TestSuite {
    let cases = vec![
        // Add edge cases that commonly cause issues
        TestCase {
            name: "edge_case_max_bounded_range".to_string(),
            description: "Maximum value in bounded range".to_string(),
            value: ChoiceValue::Integer(100),
            constraints: Constraints::Integer(IntegerConstraints {
                min_value: Some(-100),
                max_value: Some(100),
                weights: None,
                shrink_towards: Some(0),
            }),
            expected_properties: vec!["roundtrip".to_string()],
        },
    ];
    
    TestSuite {
        name: "Edge Cases".to_string(),
        cases,
    }
}