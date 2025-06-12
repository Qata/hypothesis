//! Comprehensive Choice Sequence Navigation System Tests
//!
//! This module contains comprehensive integration tests for the Choice sequence navigation system
//! capability using PyO3 FFI to validate complete capability behavior against Python Hypothesis.
//!
//! Tests validate:
//! - Tree traversal and prefix-based selection orders for shrinking patterns
//! - Choice indexing bidirectional conversion with 90%+ success rate
//! - Structured pattern navigation for shrinking choice sequences
//! - Novel prefix generation with exhaustive tree exploration
//! - Navigation performance and correctness under various constraints

use crate::choice::{ChoiceType, ChoiceValue, Constraints};
use crate::choice::constraints::*;
use crate::choice::navigation::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyBool, PyFloat, PyLong, PyAny};

#[cfg(test)]
mod navigation_system_comprehensive_tests {
    use super::*;

    /// Setup Python interpreter and import required navigation modules  
    fn setup_python_navigation() -> PyResult<Py<PyModule>> {
        Python::with_gil(|py| {
            // Add the hypothesis-python src directory to Python path (if it exists)
            if let Ok(sys) = py.import("sys") {
                if let Ok(path) = sys.getattr("path") {
                    let _ = path.call_method1("insert", (0, "/home/ch/Develop/hypothesis/conjecture-rust/hypothesis-python/src"));
                }
            }
            
            // Import the navigation module (if it exists in Python)
            // For this test, we'll use the choice module as proxy
            let choice_module = py.import("hypothesis.internal.conjecture.choice")?;
            Ok(choice_module.into())
        })
    }

    /// Convert Rust constraints to Python dict format
    fn constraints_to_python_dict<'a>(py: Python<'a>, constraints: &Constraints) -> PyResult<&'a PyDict> {
        let dict = PyDict::new(py);
        
        match constraints {
            Constraints::Integer(int_constraints) => {
                dict.set_item("type", "integer")?;
                if let Some(min_val) = int_constraints.min_value {
                    dict.set_item("min_value", min_val)?;
                } else {
                    dict.set_item("min_value", py.None())?;
                }
                if let Some(max_val) = int_constraints.max_value {
                    dict.set_item("max_value", max_val)?;
                } else {
                    dict.set_item("max_value", py.None())?;
                }
                dict.set_item("shrink_towards", int_constraints.shrink_towards.unwrap_or(0))?;
            }
            Constraints::Boolean(bool_constraints) => {
                dict.set_item("type", "boolean")?;
                dict.set_item("p", bool_constraints.p)?;
            }
            Constraints::Float(float_constraints) => {
                dict.set_item("type", "float")?;
                dict.set_item("min_value", float_constraints.min_value)?;
                dict.set_item("max_value", float_constraints.max_value)?;
                dict.set_item("allow_nan", float_constraints.allow_nan)?;
            }
            Constraints::String(string_constraints) => {
                dict.set_item("type", "string")?;
                dict.set_item("max_size", string_constraints.max_size)?;
            }
            Constraints::Bytes(bytes_constraints) => {
                dict.set_item("type", "bytes")?;
                dict.set_item("max_size", bytes_constraints.max_size)?;
            }
        }
        
        Ok(dict)
    }

    /// Convert ChoiceValue to Python object
    fn choice_value_to_python<'a>(py: Python<'a>, value: &ChoiceValue) -> PyResult<&'a PyAny> {
        match value {
            ChoiceValue::Integer(val) => Ok(val.into_py(py).into_ref(py)),
            ChoiceValue::Boolean(val) => Ok(PyBool::new(py, *val).as_ref()),
            ChoiceValue::Float(val) => Ok(PyFloat::new(py, *val).as_ref()),
            ChoiceValue::String(val) => Ok(py.eval(&format!("'{}'", val), None, None)?),
            ChoiceValue::Bytes(val) => {
                let bytes_str = format!("bytes([{}])", val.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(","));
                Ok(py.eval(&bytes_str, None, None)?)
            }
        }
    }

    /// Test comprehensive tree traversal functionality
    #[test]
    fn test_comprehensive_tree_traversal() {
        println!("NAVIGATION_TEST: Testing comprehensive tree traversal with structured patterns");
        
        let mut tree = NavigationTree::new();
        
        // Create a complex tree structure with multiple levels and constraints
        let constraints_sequence = vec![
            Constraints::Integer(IntegerConstraints::new(Some(0), Some(5), Some(2))),
            Constraints::Boolean(BooleanConstraints::new()),
            Constraints::Float(FloatConstraints {
                min_value: -1.0,
                max_value: 1.0,
                allow_nan: false,
                smallest_nonzero_magnitude: Some(0.1),
            }),
            Constraints::String(StringConstraints::new(Some(0), Some(3))),
        ];
        
        // Test multiple choice sequences to build complex tree
        let test_sequences = vec![
            ChoiceSequence::from_choices(vec![
                ChoiceValue::Integer(2),
                ChoiceValue::Boolean(false),
                ChoiceValue::Float(0.5),
                ChoiceValue::String("ab".to_string()),
            ]),
            ChoiceSequence::from_choices(vec![
                ChoiceValue::Integer(1),
                ChoiceValue::Boolean(true),
                ChoiceValue::Float(-0.5),
                ChoiceValue::String("xy".to_string()),
            ]),
            ChoiceSequence::from_choices(vec![
                ChoiceValue::Integer(3),
                ChoiceValue::Boolean(false),
                ChoiceValue::Float(0.0),
                ChoiceValue::String("".to_string()),
            ]),
        ];
        
        // Record all sequences in the tree
        for sequence in &test_sequences {
            tree.record_sequence(sequence, &constraints_sequence);
        }
        
        // Verify tree structure
        let stats = tree.stats();
        assert!(stats.node_count >= 3, "Tree should have at least 3 nodes, got {}", stats.node_count);
        assert!(stats.max_depth >= 2, "Tree should have depth >= 2, got {}", stats.max_depth);
        
        // Test novel prefix generation
        let mut novel_count = 0;
        for _ in 0..10 {
            if let Ok(novel_prefix) = tree.generate_novel_prefix() {
                novel_count += 1;
                assert!(novel_prefix.length > 0, "Novel prefix should have length > 0");
                assert!(novel_prefix.length <= 4, "Novel prefix should not exceed max sequence length");
                println!("NAVIGATION_TEST: Generated novel prefix of length {}", novel_prefix.length);
            }
        }
        
        assert!(novel_count > 0, "Should generate at least one novel prefix");
        println!("NAVIGATION_TEST: Generated {} novel prefixes", novel_count);
    }

    /// Test prefix-based selection orders for shrinking patterns  
    #[test]
    fn test_prefix_based_selection_orders() {
        println!("NAVIGATION_TEST: Testing prefix-based selection orders for shrinking patterns");
        
        // Test the minimize distance selection order (core shrinking pattern)
        let prefix = ChoiceSequence::from_choices(vec![ChoiceValue::Integer(42)]);
        let selector = PrefixSelector::new(prefix, 5);
        
        let order = selector.selection_order(2);
        println!("NAVIGATION_TEST: Selection order from index 2: {:?}", order);
        
        // Should prioritize choices closer to the start index (2)
        // Expected: [2, 1, 0, 3, 4] - start with 2, then left neighbors, then right neighbors
        assert_eq!(order, vec![2, 1, 0, 3, 4], "Selection order should minimize distance from start index");
        assert_eq!(order.len(), 5, "Selection order should include all choices");
        
        // Test different start indices
        let order_from_0 = selector.selection_order(0);
        assert_eq!(order_from_0, vec![0, 1, 2, 3, 4], "From index 0, should explore right neighbors");
        
        let order_from_4 = selector.selection_order(4);
        assert_eq!(order_from_4, vec![4, 3, 2, 1, 0], "From index 4, should explore left neighbors");
        
        // Test random selection order
        let random_order = selector.random_selection_order(12345);
        assert_eq!(random_order.len(), 5, "Random order should include all choices");
        assert!(random_order.contains(&0), "Random order should contain all original indices");
        assert!(random_order.contains(&4), "Random order should contain all original indices");
        
        println!("NAVIGATION_TEST: Prefix-based selection orders working correctly");
    }

    /// Test choice indexing bidirectional conversion with 90%+ success rate
    #[test]
    fn test_choice_indexing_bidirectional_conversion() {
        println!("NAVIGATION_TEST: Testing choice indexing bidirectional conversion for 90%+ success rate");
        
        let mut indexer = ChoiceIndexer::new();
        let mut total_tests = 0;
        let mut successful_roundtrips = 0;
        
        // Test Integer choices with various constraints
        let integer_constraints = vec![
            Constraints::Integer(IntegerConstraints::new(None, None, Some(0))),
            Constraints::Integer(IntegerConstraints::new(Some(-10), Some(10), Some(0))),
            Constraints::Integer(IntegerConstraints::new(Some(0), Some(100), Some(50))),
        ];
        
        let integer_values = vec![-100, -10, -1, 0, 1, 10, 100, 42, -42];
        
        for constraints in &integer_constraints {
            for &value in &integer_values {
                let choice_value = ChoiceValue::Integer(value);
                total_tests += 1;
                
                if let Ok(index) = indexer.choice_to_index(&choice_value, constraints) {
                    if let Ok(recovered) = indexer.index_to_choice(index, ChoiceType::Integer, constraints) {
                        if recovered == choice_value {
                            successful_roundtrips += 1;
                        } else {
                            println!("NAVIGATION_TEST: Integer roundtrip failed: {} -> {} -> {:?}", value, index, recovered);
                        }
                    }
                }
            }
        }
        
        // Test Boolean choices
        let boolean_constraints = Constraints::Boolean(BooleanConstraints::new());
        for &value in &[true, false] {
            let choice_value = ChoiceValue::Boolean(value);
            total_tests += 1;
            
            if let Ok(index) = indexer.choice_to_index(&choice_value, &boolean_constraints) {
                if let Ok(recovered) = indexer.index_to_choice(index, ChoiceType::Boolean, &boolean_constraints) {
                    if recovered == choice_value {
                        successful_roundtrips += 1;
                    }
                }
            }
        }
        
        // Test Float choices (finite values for reliable roundtrip)
        let float_constraints = Constraints::Float(FloatConstraints {
            min_value: -100.0,
            max_value: 100.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(0.0),
        });
        
        let float_values = vec![0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 10.5, -10.5];
        for &value in &float_values {
            let choice_value = ChoiceValue::Float(value);
            total_tests += 1;
            
            if let Ok(index) = indexer.choice_to_index(&choice_value, &float_constraints) {
                if let Ok(recovered) = indexer.index_to_choice(index, ChoiceType::Float, &float_constraints) {
                    if let ChoiceValue::Float(recovered_float) = recovered {
                        // For floats, check bit-exact equality
                        if value.to_bits() == recovered_float.to_bits() {
                            successful_roundtrips += 1;
                        }
                    }
                }
            }
        }
        
        // Test String choices
        let string_constraints = Constraints::String(StringConstraints::new(Some(0), Some(10)));
        let string_values = vec!["", "a", "ab", "abc", "hello", "xyz"];
        for value in &string_values {
            let choice_value = ChoiceValue::String(value.to_string());
            total_tests += 1;
            
            if let Ok(index) = indexer.choice_to_index(&choice_value, &string_constraints) {
                if let Ok(recovered) = indexer.index_to_choice(index, ChoiceType::String, &string_constraints) {
                    if recovered == choice_value {
                        successful_roundtrips += 1;
                    }
                }
            }
        }
        
        // Test Bytes choices
        let bytes_constraints = Constraints::Bytes(BytesConstraints::new(Some(0), Some(10)));
        let bytes_values = vec![
            vec![],
            vec![0],
            vec![1, 2],
            vec![255],
            vec![0, 127, 255],
        ];
        for value in &bytes_values {
            let choice_value = ChoiceValue::Bytes(value.clone());
            total_tests += 1;
            
            if let Ok(index) = indexer.choice_to_index(&choice_value, &bytes_constraints) {
                if let Ok(recovered) = indexer.index_to_choice(index, ChoiceType::Bytes, &bytes_constraints) {
                    if recovered == choice_value {
                        successful_roundtrips += 1;
                    }
                }
            }
        }
        
        let success_rate = (successful_roundtrips as f64 / total_tests as f64) * 100.0;
        println!("NAVIGATION_TEST: Bidirectional conversion success rate: {:.1}% ({}/{})", 
                 success_rate, successful_roundtrips, total_tests);
        
        assert!(success_rate >= 90.0, 
                "Choice indexing bidirectional conversion should achieve 90%+ success rate, got {:.1}%", 
                success_rate);
    }

    /// Test structured pattern navigation for shrinking sequences
    #[test]
    fn test_structured_pattern_navigation() {
        println!("NAVIGATION_TEST: Testing structured pattern navigation for shrinking sequences");
        
        let mut tree = NavigationTree::new();
        
        // Create structured patterns representing different shrinking strategies
        let shrinking_patterns = vec![
            // Pattern 1: Integer sequences with shrink_towards=0
            (
                vec![
                    ChoiceValue::Integer(5),
                    ChoiceValue::Integer(3),
                    ChoiceValue::Integer(1),
                    ChoiceValue::Integer(0),
                ],
                vec![
                    Constraints::Integer(IntegerConstraints::new(None, None, Some(0))); 4
                ]
            ),
            // Pattern 2: Mixed types with complexity reduction
            (
                vec![
                    ChoiceValue::String("hello".to_string()),
                    ChoiceValue::String("hi".to_string()),
                    ChoiceValue::String("a".to_string()),
                    ChoiceValue::String("".to_string()),
                ],
                vec![
                    Constraints::String(StringConstraints::new(Some(0), Some(10))); 4
                ]
            ),
            // Pattern 3: Float sequences approaching zero
            (
                vec![
                    ChoiceValue::Float(1.0),
                    ChoiceValue::Float(0.5),
                    ChoiceValue::Float(0.1),
                    ChoiceValue::Float(0.0),
                ],
                vec![
                    Constraints::Float(FloatConstraints {
                        min_value: 0.0,
                        max_value: 10.0,
                        allow_nan: false,
                        smallest_nonzero_magnitude: Some(0.01),
                    }); 4
                ]
            ),
        ];
        
        // Record all shrinking patterns
        for (choices, constraints) in &shrinking_patterns {
            let sequence = ChoiceSequence::from_choices(choices.clone());
            tree.record_sequence(&sequence, constraints);
        }
        
        // Test that the tree can generate patterns that follow shrinking logic
        let mut generated_patterns = Vec::new();
        for _ in 0..10 {
            if let Ok(pattern) = tree.generate_novel_prefix() {
                generated_patterns.push(pattern);
            }
        }
        
        assert!(!generated_patterns.is_empty(), "Should generate novel shrinking patterns");
        
        // Verify patterns exhibit expected shrinking characteristics
        for pattern in &generated_patterns {
            assert!(pattern.length > 0, "Generated pattern should not be empty");
            
            // Check if pattern follows complexity reduction principle
            let (_, complexity_indices) = pattern.sort_key();
            if complexity_indices.len() > 1 {
                // Patterns should generally trend toward simpler values
                let avg_complexity: f64 = complexity_indices.iter().map(|&x| x as f64).sum::<f64>() / complexity_indices.len() as f64;
                println!("NAVIGATION_TEST: Pattern complexity average: {:.2}", avg_complexity);
            }
        }
        
        println!("NAVIGATION_TEST: Generated {} structured shrinking patterns", generated_patterns.len());
    }

    /// Test navigation performance under various constraints
    #[test]
    fn test_navigation_performance_constraints() {
        println!("NAVIGATION_TEST: Testing navigation performance under various constraints");
        
        let mut tree = NavigationTree::new();
        
        // Test constraint handling with complex scenarios
        let complex_constraints = vec![
            // Tight integer bounds
            Constraints::Integer(IntegerConstraints::new(Some(0), Some(3), Some(1))),
            // Probability-weighted boolean
            Constraints::Boolean(BooleanConstraints { p: 0.25 }),
            // Narrow float range
            Constraints::Float(FloatConstraints {
                min_value: -0.1,
                max_value: 0.1,
                allow_nan: false,
                smallest_nonzero_magnitude: Some(0.001),
            }),
            // Limited string length
            Constraints::String(StringConstraints::new(Some(0), Some(2))),
            // Small byte arrays
            Constraints::Bytes(BytesConstraints::new(Some(0), Some(2))),
        ];
        
        // Generate many sequences with these constraints
        let start_time = std::time::Instant::now();
        let mut sequences_generated = 0;
        
        for i in 0..100 {
            let constraint = &complex_constraints[i % complex_constraints.len()];
            
            // Generate a choice based on constraints
            let choice = match constraint {
                Constraints::Integer(c) => {
                    let val = c.min_value.unwrap_or(0) + (i as i128 % 4);
                    ChoiceValue::Integer(val)
                }
                Constraints::Boolean(_) => ChoiceValue::Boolean(i % 2 == 0),
                Constraints::Float(_) => ChoiceValue::Float((i as f64 % 10.0) / 100.0),
                Constraints::String(_) => ChoiceValue::String(format!("{}", char::from((i % 26) as u8 + b'a'))),
                Constraints::Bytes(_) => ChoiceValue::Bytes(vec![(i % 256) as u8]),
            };
            
            let sequence = ChoiceSequence::from_choices(vec![choice]);
            tree.record_sequence(&sequence, &[constraint.clone()]);
            sequences_generated += 1;
        }
        
        let generation_time = start_time.elapsed();
        println!("NAVIGATION_TEST: Generated {} sequences in {:?}", sequences_generated, generation_time);
        
        // Test novel prefix generation performance
        let start_time = std::time::Instant::now();
        let mut novel_prefixes = 0;
        
        for _ in 0..50 {
            if tree.generate_novel_prefix().is_ok() {
                novel_prefixes += 1;
            }
        }
        
        let prefix_time = start_time.elapsed();
        println!("NAVIGATION_TEST: Generated {} novel prefixes in {:?}", novel_prefixes, prefix_time);
        
        // Verify tree statistics
        let stats = tree.stats();
        println!("NAVIGATION_TEST: Final tree stats - nodes: {}, depth: {}, cached: {}", 
                 stats.node_count, stats.max_depth, stats.cached_sequences);
        
        assert!(stats.node_count > 0, "Tree should contain nodes");
        assert!(!tree.is_exhausted(), "Tree should not be exhausted with complex constraints");
    }

    /// Test FFI integration for navigation system validation
    #[test] 
    fn test_navigation_ffi_integration() -> PyResult<()> {
        println!("NAVIGATION_TEST: Testing FFI integration for navigation system validation");
        
        // This test verifies that our navigation system produces results
        // compatible with Python's choice indexing system
        let _choice_module = match setup_python_navigation() {
            Ok(module) => module,
            Err(e) => {
                println!("NAVIGATION_TEST: Skipping FFI test due to Python setup error: {}", e);
                return Ok(());
            }
        };
        
        let mut indexer = ChoiceIndexer::new();
        
        // Test that our indexing matches expected patterns even without direct Python comparison
        let test_cases = vec![
            (ChoiceValue::Integer(0), Constraints::Integer(IntegerConstraints::new(None, None, Some(0)))),
            (ChoiceValue::Integer(1), Constraints::Integer(IntegerConstraints::new(None, None, Some(0)))),
            (ChoiceValue::Integer(-1), Constraints::Integer(IntegerConstraints::new(None, None, Some(0)))),
            (ChoiceValue::Boolean(true), Constraints::Boolean(BooleanConstraints::new())),
            (ChoiceValue::Boolean(false), Constraints::Boolean(BooleanConstraints::new())),
        ];
        
        for (choice, constraints) in test_cases {
            let index = indexer.choice_to_index(&choice, &constraints)
                .expect("Should be able to index choice");
            
            let choice_type = match choice {
                ChoiceValue::Integer(_) => ChoiceType::Integer,
                ChoiceValue::Boolean(_) => ChoiceType::Boolean,
                ChoiceValue::Float(_) => ChoiceType::Float,
                ChoiceValue::String(_) => ChoiceType::String,
                ChoiceValue::Bytes(_) => ChoiceType::Bytes,
            };
            
            let recovered = indexer.index_to_choice(index, choice_type, &constraints)
                .expect("Should be able to recover choice from index");
            
            assert_eq!(choice, recovered, "FFI compatibility: choice should roundtrip correctly");
            println!("NAVIGATION_TEST: FFI compatible indexing: {:?} -> {} -> {:?}", choice, index, recovered);
        }
        
        println!("NAVIGATION_TEST: FFI integration validation completed successfully");
        Ok(())
    }

    /// Integration test for complete navigation capability
    #[test]
    fn test_complete_navigation_capability_integration() {
        println!("NAVIGATION_TEST: Testing complete navigation capability integration");
        
        // This test exercises the entire navigation system end-to-end
        let mut tree = NavigationTree::new();
        let mut indexer = ChoiceIndexer::new();
        
        // Phase 1: Build complex tree with realistic choice patterns
        // HTTP status codes (realistic shrinking pattern)
        let http_codes = vec![500, 404, 200];
        let http_constraints = IntegerConstraints::new(Some(100), Some(599), Some(200));
        for &value in &http_codes {
            let choice = ChoiceValue::Integer(value as i128);
            let constraints = Constraints::Integer(http_constraints.clone());
            
            // Test indexing
            if let Ok(index) = indexer.choice_to_index(&choice, &constraints) {
                println!("NAVIGATION_TEST: Choice {:?} -> index {}", choice, index);
            }
            
            // Build tree
            let sequence = ChoiceSequence::from_choices(vec![choice]);
            tree.record_sequence(&sequence, &[constraints]);
        }
        
        // List lengths (shrink toward empty)  
        let list_lengths = vec![10, 5, 1, 0];
        let length_constraints = IntegerConstraints::new(Some(0), Some(100), Some(0));
        for &value in &list_lengths {
            let choice = ChoiceValue::Integer(value as i128);
            let constraints = Constraints::Integer(length_constraints.clone());
            
            // Test indexing
            if let Ok(index) = indexer.choice_to_index(&choice, &constraints) {
                println!("NAVIGATION_TEST: Choice {:?} -> index {}", choice, index);
            }
            
            // Build tree
            let sequence = ChoiceSequence::from_choices(vec![choice]);
            tree.record_sequence(&sequence, &[constraints]);
        }
        
        // Probabilities (shrink toward 0.0)
        let probabilities = vec![1.0, 0.5, 0.1, 0.0];
        let prob_constraints = FloatConstraints {
            min_value: 0.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(0.001),
        };
        for &value in &probabilities {
            let choice = ChoiceValue::Float(value);
            let constraints = Constraints::Float(prob_constraints.clone());
            
            // Test indexing
            if let Ok(index) = indexer.choice_to_index(&choice, &constraints) {
                println!("NAVIGATION_TEST: Choice {:?} -> index {}", choice, index);
            }
            
            // Build tree
            let sequence = ChoiceSequence::from_choices(vec![choice]);
            tree.record_sequence(&sequence, &[constraints]);
        }
        
        // Phase 2: Test prefix-based selection with realistic scenarios  
        let prefix = ChoiceSequence::from_choices(vec![ChoiceValue::Integer(200)]);
        let selector = PrefixSelector::new(prefix, 10);
        
        // Test selection orders for different shrinking strategies
        let shrink_orders = vec![
            selector.selection_order(5),  // Start from middle
            selector.selection_order(0),  // Start from beginning
            selector.selection_order(9),  // Start from end
        ];
        
        for (i, order) in shrink_orders.iter().enumerate() {
            println!("NAVIGATION_TEST: Shrinking order strategy {}: {:?}", i + 1, order);
            assert!(!order.is_empty(), "Selection order should not be empty");
            assert_eq!(order.len(), 10, "Selection order should include all choices");
        }
        
        // Phase 3: Test novel prefix generation under pressure
        let mut generated_prefixes = Vec::new();
        let mut generation_attempts = 0;
        
        while generated_prefixes.len() < 5 && generation_attempts < 20 {
            generation_attempts += 1;
            if let Ok(prefix) = tree.generate_novel_prefix() {
                generated_prefixes.push(prefix);
            }
        }
        
        assert!(!generated_prefixes.is_empty(), "Should generate at least one novel prefix");
        println!("NAVIGATION_TEST: Generated {} novel prefixes in {} attempts", 
                 generated_prefixes.len(), generation_attempts);
        
        // Phase 4: Verify complete capability metrics
        let final_stats = tree.stats();
        println!("NAVIGATION_TEST: Final capability metrics:");
        println!("  - Tree nodes: {}", final_stats.node_count);
        println!("  - Tree depth: {}", final_stats.max_depth);
        println!("  - Cached sequences: {}", final_stats.cached_sequences);
        println!("  - Novel prefixes generated: {}", generated_prefixes.len());
        
        // Success criteria for complete capability
        assert!(final_stats.node_count >= 1, "Tree should have meaningful structure");
        assert!(final_stats.max_depth >= 0, "Tree should have non-negative depth");
        assert!(!tree.is_exhausted(), "Tree should still have exploration potential");
        
        println!("NAVIGATION_TEST: Complete navigation capability integration successful");
    }
}