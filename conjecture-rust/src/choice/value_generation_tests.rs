//! Integration tests for the Choice Value Generation System
//!
//! These tests verify that the value generation system correctly integrates
//! with the existing choice system and produces values that match Python
//! Hypothesis behavior patterns.

#[cfg(test)]
mod integration_tests {
    use crate::choice::{
        ChoiceType, ChoiceValue, Constraints,
        IntegerConstraints, BooleanConstraints, FloatConstraints,
        StringConstraints, BytesConstraints, IntervalSet,
        StandardValueGenerator, ValueGenerator,
        BufferEntropySource,
        choice_permitted, choice_equal
    };
    use std::collections::{HashMap, HashSet};

    /// Test that generated values satisfy their constraints
    #[test]
    fn test_constraint_satisfaction() {
        let mut generator = StandardValueGenerator::new();
        
        // Test integer constraints
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(-10),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        });
        
        for entropy_val in 0..=255u8 {
            let mut entropy = BufferEntropySource::new(vec![entropy_val, entropy_val, entropy_val, entropy_val]);
            let value = generator.generate_value(ChoiceType::Integer, &int_constraints, &mut entropy).unwrap();
            
            assert!(choice_permitted(&value, &int_constraints), 
                    "Generated value {:?} violates constraints with entropy {}", value, entropy_val);
        }
        
        // Test boolean constraints  
        let bool_constraints = Constraints::Boolean(BooleanConstraints { p: 0.7 });
        
        for entropy_val in 0..=255u8 {
            let mut entropy = BufferEntropySource::new(vec![entropy_val]);
            let value = generator.generate_value(ChoiceType::Boolean, &bool_constraints, &mut entropy).unwrap();
            
            assert!(choice_permitted(&value, &bool_constraints),
                    "Generated boolean {:?} violates constraints with entropy {}", value, entropy_val);
        }
    }
    
    /// Test distribution properties of generated values
    #[test]
    fn test_value_distributions() {
        let mut generator = StandardValueGenerator::new();
        
        // Test boolean distribution
        let bool_constraints = Constraints::Boolean(BooleanConstraints { p: 0.3 });
        let mut true_count = 0;
        let total_samples = 1000;
        
        for i in 0..total_samples {
            let entropy_byte = (i * 7) % 256; // Pseudo-random pattern
            let mut entropy = BufferEntropySource::new(vec![entropy_byte as u8]);
            
            if let ChoiceValue::Boolean(value) = generator.generate_value(ChoiceType::Boolean, &bool_constraints, &mut entropy).unwrap() {
                if value {
                    true_count += 1;
                }
            }
        }
        
        let true_ratio = true_count as f64 / total_samples as f64;
        
        // Should be approximately 0.3 with some tolerance
        assert!((true_ratio - 0.3).abs() < 0.1, 
                "Boolean distribution ratio {:.3} too far from expected 0.3", true_ratio);
    }
    
    /// Test integer range distribution
    #[test]
    fn test_integer_range_distribution() {
        let mut generator = StandardValueGenerator::new();
        
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(9), // Range [0, 9]
            weights: None,
            shrink_towards: Some(0),
        });
        
        let mut counts = HashMap::new();
        let total_samples = 1000;
        
        for i in 0..total_samples {
            let entropy_bytes = vec![(i % 256) as u8, ((i * 3) % 256) as u8];
            let mut entropy = BufferEntropySource::new(entropy_bytes);
            
            if let ChoiceValue::Integer(value) = generator.generate_value(ChoiceType::Integer, &constraints, &mut entropy).unwrap() {
                *counts.entry(value).or_insert(0) += 1;
            }
        }
        
        // Should have reasonable distribution across the range
        assert!(counts.len() >= 5, "Too few unique values generated: {}", counts.len());
        
        for (&value, &count) in &counts {
            assert!(value >= 0 && value <= 9, "Value {} outside expected range", value);
            assert!(count > 0, "Zero count for value {}", value);
        }
    }
    
    /// Test weighted integer selection
    #[test]
    fn test_weighted_integer_selection() {
        let mut generator = StandardValueGenerator::new();
        
        let mut weights = HashMap::new();
        weights.insert(1, 0.1); // Low weight
        weights.insert(2, 0.8); // High weight
        weights.insert(3, 0.1); // Low weight
        
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(1),
            max_value: Some(3),
            weights: Some(weights),
            shrink_towards: Some(2),
        });
        
        let mut counts = HashMap::new();
        let total_samples = 1000;
        
        for i in 0..total_samples {
            let entropy_bytes = vec![
                (i % 256) as u8, 
                ((i * 7) % 256) as u8,
                ((i * 13) % 256) as u8,
                ((i * 19) % 256) as u8,
            ];
            let mut entropy = BufferEntropySource::new(entropy_bytes);
            
            if let ChoiceValue::Integer(value) = generator.generate_value(ChoiceType::Integer, &constraints, &mut entropy).unwrap() {
                *counts.entry(value).or_insert(0) += 1;
            }
        }
        
        // Value 2 should be much more common than 1 or 3
        let count_1 = counts.get(&1).unwrap_or(&0);
        let count_2 = counts.get(&2).unwrap_or(&0);
        let count_3 = counts.get(&3).unwrap_or(&0);
        
        assert!(*count_2 > total_samples / 2, "Value 2 should dominate with count {} > {}", count_2, total_samples / 2);
        assert!(*count_2 > *count_1 * 3, "Value 2 count {} should be >> value 1 count {}", count_2, count_1);
        assert!(*count_2 > *count_3 * 3, "Value 2 count {} should be >> value 3 count {}", count_2, count_3);
    }
    
    /// Test string generation with character intervals
    #[test]
    fn test_string_character_intervals() {
        let mut generator = StandardValueGenerator::new();
        
        let constraints = Constraints::String(StringConstraints {
            min_size: 5,
            max_size: 5, // Fixed length
            intervals: IntervalSet::from_string("ABCDEF"),
        });
        
        let mut all_chars = HashSet::new();
        
        for i in 0..100 {
            let entropy_bytes = (0..20).map(|j| ((i * 7 + j * 11) % 256) as u8).collect(); // More entropy for string generation
            let mut entropy = BufferEntropySource::new(entropy_bytes);
            
            if let ChoiceValue::String(value) = generator.generate_value(ChoiceType::String, &constraints, &mut entropy).unwrap() {
                assert_eq!(value.len(), 5, "String length should be 5, got {}", value.len());
                
                for ch in value.chars() {
                    assert!("ABCDEF".contains(ch), "Character '{}' not in allowed set", ch);
                    all_chars.insert(ch);
                }
            }
        }
        
        // Should have generated variety of allowed characters
        assert!(all_chars.len() >= 3, "Should generate variety of characters, got {:?}", all_chars);
    }
    
    /// Test bytes generation
    #[test]
    fn test_bytes_generation() {
        let mut generator = StandardValueGenerator::new();
        
        let constraints = Constraints::Bytes(BytesConstraints {
            min_size: 3,
            max_size: 7,
        });
        
        let mut length_counts = HashMap::new();
        
        for i in 0..200 {
            let entropy_bytes = (0..10).map(|j| ((i * 3 + j * 5) % 256) as u8).collect();
            let mut entropy = BufferEntropySource::new(entropy_bytes);
            
            if let ChoiceValue::Bytes(value) = generator.generate_value(ChoiceType::Bytes, &constraints, &mut entropy).unwrap() {
                assert!(value.len() >= 3 && value.len() <= 7, 
                        "Bytes length {} outside range [3, 7]", value.len());
                
                *length_counts.entry(value.len()).or_insert(0) += 1;
            }
        }
        
        // Should have variety of lengths
        assert!(length_counts.len() >= 3, "Should generate various lengths, got {:?}", length_counts);
    }
    
    /// Test float constraint handling
    #[test]
    fn test_float_constraint_handling() {
        let mut generator = StandardValueGenerator::new();
        
        let constraints = Constraints::Float(FloatConstraints {
            min_value: -5.0,
            max_value: 5.0,
            allow_nan: false,
            smallest_nonzero_magnitude: 0.01,
        });
        
        for i in 0..100 {
            let entropy_bytes = (0..8).map(|j| ((i * 7 + j * 11) % 256) as u8).collect();
            let mut entropy = BufferEntropySource::new(entropy_bytes);
            
            if let ChoiceValue::Float(value) = generator.generate_value(ChoiceType::Float, &constraints, &mut entropy).unwrap() {
                assert!(!value.is_nan(), "NaN should not be generated when disallowed");
                assert!(value >= -5.0 && value <= 5.0, "Float {} outside range [-5.0, 5.0]", value);
                
                if value != 0.0 {
                    assert!(value.abs() >= 0.01, "Non-zero float {} below minimum magnitude 0.01", value);
                }
            }
        }
    }
    
    /// Test that generated values are deterministic for same entropy
    #[test]
    fn test_deterministic_generation() {
        let mut generator1 = StandardValueGenerator::new();
        let mut generator2 = StandardValueGenerator::new();
        
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });
        
        let entropy_data = vec![0x12, 0x34, 0x56, 0x78];
        
        let mut entropy1 = BufferEntropySource::new(entropy_data.clone());
        let mut entropy2 = BufferEntropySource::new(entropy_data);
        
        let value1 = generator1.generate_value(ChoiceType::Integer, &constraints, &mut entropy1).unwrap();
        let value2 = generator2.generate_value(ChoiceType::Integer, &constraints, &mut entropy2).unwrap();
        
        assert!(choice_equal(&value1, &value2), 
                "Same entropy should produce same values: {:?} vs {:?}", value1, value2);
    }
    
    /// Test error handling for insufficient entropy
    #[test]
    fn test_insufficient_entropy_handling() {
        let mut generator = StandardValueGenerator::new();
        
        let constraints = Constraints::String(StringConstraints {
            min_size: 10,
            max_size: 10,
            intervals: IntervalSet::from_string("ABC"),
        });
        
        let mut entropy = BufferEntropySource::new(vec![0x12]); // Only 1 byte, need more
        
        let result = generator.generate_value(ChoiceType::String, &constraints, &mut entropy);
        assert!(result.is_err(), "Should fail with insufficient entropy");
    }
    
    /// Test constraint validation during generation
    #[test]  
    fn test_constraint_validation() {
        let mut generator = StandardValueGenerator::new();
        
        // Test mismatched type and constraints
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        });
        
        let mut entropy = BufferEntropySource::new(vec![0x12]);
        
        let result = generator.generate_value(ChoiceType::Boolean, &int_constraints, &mut entropy);
        assert!(result.is_err(), "Should fail with mismatched types");
    }
    
    /// Test integration with choice_permitted function
    #[test]
    fn test_choice_permitted_integration() {
        let mut generator = StandardValueGenerator::new();
        
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(5),
            max_value: Some(15),
            weights: None,
            shrink_towards: Some(10),
        });
        
        // Generate many values and verify all are permitted
        for entropy_val in 0..=255u8 {
            let mut entropy = BufferEntropySource::new(vec![entropy_val, entropy_val]);
            
            let value = generator.generate_value(ChoiceType::Integer, &constraints, &mut entropy).unwrap();
            
            assert!(choice_permitted(&value, &constraints),
                    "Generated value {:?} not permitted by constraints (entropy: {})", 
                    value, entropy_val);
        }
    }
    
    /// Performance test to ensure reasonable generation speed
    #[test]
    fn test_generation_performance() {
        let mut generator = StandardValueGenerator::new();
        
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(1000),
            weights: None,
            shrink_towards: Some(0),
        });
        
        let start = std::time::Instant::now();
        
        // Generate 10000 values
        for i in 0..10000 {
            let entropy_bytes = vec![(i % 256) as u8, ((i / 256) % 256) as u8];
            let mut entropy = BufferEntropySource::new(entropy_bytes);
            
            let _value = generator.generate_value(ChoiceType::Integer, &constraints, &mut entropy).unwrap();
        }
        
        let duration = start.elapsed();
        
        // Should complete in reasonable time (less than 1 second for 10k generations)
        assert!(duration.as_millis() < 1000, 
                "Generation took too long: {}ms for 10k values", duration.as_millis());
    }
}