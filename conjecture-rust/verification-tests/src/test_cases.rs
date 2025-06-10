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
                smallest_nonzero_magnitude: None,
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
                smallest_nonzero_magnitude: None,
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
                smallest_nonzero_magnitude: None,
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
                smallest_nonzero_magnitude: None,
            }),
            expected_properties: vec!["sign_bit_one".to_string(), "roundtrip".to_string()],
        },
    ];
    
    TestSuite {
        name: "Float Choice Indexing".to_string(),
        cases,
    }
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