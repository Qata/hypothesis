//! Value handling and utility functions for choices

use super::{ChoiceValue, Constraints};

/// Check if two choice values are equal (handles special float cases)
pub fn choice_equal(a: &ChoiceValue, b: &ChoiceValue) -> bool {
    
    match (a, b) {
        (ChoiceValue::Integer(a), ChoiceValue::Integer(b)) => {
            a == b
        }
        (ChoiceValue::Boolean(a), ChoiceValue::Boolean(b)) => {
            a == b
        }
        (ChoiceValue::Float(a), ChoiceValue::Float(b)) => {
            // Handle NaN and -0.0/+0.0 cases
            let result = if a.is_nan() && b.is_nan() {
                true
            } else if a.is_nan() || b.is_nan() {
                false
            } else {
                // Use bitwise comparison to distinguish -0.0 and +0.0
                a.to_bits() == b.to_bits()
            };
            result
        }
        (ChoiceValue::String(a), ChoiceValue::String(b)) => {
            a == b
        }
        (ChoiceValue::Bytes(a), ChoiceValue::Bytes(b)) => {
            a == b
        }
        _ => {
            false
        }
    }
}

/// Check if a choice value is permitted under the given constraints
pub fn choice_permitted(value: &ChoiceValue, constraints: &Constraints) -> bool {
    
    match (value, constraints) {
        (ChoiceValue::Integer(val), Constraints::Integer(c)) => {
            
            if let Some(min) = c.min_value {
                if *val < min {
                    return false;
                }
            }
            
            if let Some(max) = c.max_value {
                if *val > max {
                    return false;
                }
            }
            
            true
        }
        
        (ChoiceValue::Boolean(val), Constraints::Boolean(c)) => {
            
            let result = if c.p == 0.0 {
                !val // Only false permitted when p=0.0
            } else if c.p == 1.0 {
                *val // Only true permitted when p=1.0
            } else {
                true // Both values permitted for 0 < p < 1
            };
            
            result
        }
        
        (ChoiceValue::Float(val), Constraints::Float(c)) => {
            
            if val.is_nan() {
                return c.allow_nan;
            }
            
            if *val < c.min_value || *val > c.max_value {
                return false;
            }
            
            let smallest = c.smallest_nonzero_magnitude;
            if smallest > 0.0 {
                let abs_val = val.abs();
                if abs_val != 0.0 && abs_val < smallest {
                    return false;
                }
            }
            
            true
        }
        
        (ChoiceValue::String(val), Constraints::String(c)) => {
            
            if val.len() < c.min_size || val.len() > c.max_size {
                return false;
            }
            
            // Check if all characters are in allowed intervals
            for ch in val.chars() {
                let code = ch as u32;
                let mut found = false;
                for (start, end) in &c.intervals.intervals {
                    if code >= *start && code <= *end {
                        found = true;
                        break;
                    }
                }
                if !found {
                    return false;
                }
            }
            
            true
        }
        
        (ChoiceValue::Bytes(val), Constraints::Bytes(c)) => {
            
            val.len() >= c.min_size && val.len() <= c.max_size
        }
        
        _ => {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints, BytesConstraints, IntervalSet};

    #[test]
    fn test_choice_equal_integers() {
        
        let a = ChoiceValue::Integer(42);
        let b = ChoiceValue::Integer(42);
        let c = ChoiceValue::Integer(43);
        
        assert!(choice_equal(&a, &b));
        assert!(!choice_equal(&a, &c));
    }

    #[test]
    fn test_choice_equal_floats() {
        
        let a = ChoiceValue::Float(1.0);
        let b = ChoiceValue::Float(1.0);
        let c = ChoiceValue::Float(2.0);
        let nan1 = ChoiceValue::Float(f64::NAN);
        let nan2 = ChoiceValue::Float(f64::NAN);
        let zero_pos = ChoiceValue::Float(0.0);
        let zero_neg = ChoiceValue::Float(-0.0);
        
        assert!(choice_equal(&a, &b));
        assert!(!choice_equal(&a, &c));
        assert!(choice_equal(&nan1, &nan2)); // NaN == NaN
        assert!(!choice_equal(&zero_pos, &zero_neg)); // +0.0 != -0.0
    }

    #[test]
    fn test_choice_permitted_integer() {
        
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        });
        
        assert!(choice_permitted(&ChoiceValue::Integer(5), &constraints));
        assert!(choice_permitted(&ChoiceValue::Integer(0), &constraints));
        assert!(choice_permitted(&ChoiceValue::Integer(10), &constraints));
        assert!(!choice_permitted(&ChoiceValue::Integer(-1), &constraints));
        assert!(!choice_permitted(&ChoiceValue::Integer(11), &constraints));
    }

    #[test]
    fn test_choice_permitted_boolean() {
        
        let constraints_zero = Constraints::Boolean(BooleanConstraints { p: 0.0 });
        let constraints_one = Constraints::Boolean(BooleanConstraints { p: 1.0 });
        let constraints_half = Constraints::Boolean(BooleanConstraints { p: 0.5 });
        
        // p = 0.0: only false permitted
        assert!(!choice_permitted(&ChoiceValue::Boolean(true), &constraints_zero));
        assert!(choice_permitted(&ChoiceValue::Boolean(false), &constraints_zero));
        
        // p = 1.0: only true permitted
        assert!(choice_permitted(&ChoiceValue::Boolean(true), &constraints_one));
        assert!(!choice_permitted(&ChoiceValue::Boolean(false), &constraints_one));
        
        // p = 0.5: both permitted
        assert!(choice_permitted(&ChoiceValue::Boolean(true), &constraints_half));
        assert!(choice_permitted(&ChoiceValue::Boolean(false), &constraints_half));
    }

    #[test]
    fn test_choice_permitted_float_nan() {
        
        let constraints_allow_nan = Constraints::Float(FloatConstraints {
            min_value: 0.0,
            max_value: 10.0,
            allow_nan: true,
            smallest_nonzero_magnitude: f64::MIN_POSITIVE,
        });
        
        let constraints_no_nan = Constraints::Float(FloatConstraints {
            min_value: 0.0,
            max_value: 10.0,
            allow_nan: false,
            smallest_nonzero_magnitude: f64::MIN_POSITIVE,
        });
        
        assert!(choice_permitted(&ChoiceValue::Float(f64::NAN), &constraints_allow_nan));
        assert!(!choice_permitted(&ChoiceValue::Float(f64::NAN), &constraints_no_nan));
    }

    #[test]
    fn test_choice_permitted_string() {
        
        let constraints = Constraints::String(StringConstraints {
            min_size: 1,
            max_size: 5,
            intervals: IntervalSet::from_string("abc"),
        });
        
        assert!(choice_permitted(&ChoiceValue::String("abc".to_string()), &constraints));
        assert!(choice_permitted(&ChoiceValue::String("a".to_string()), &constraints));
        assert!(!choice_permitted(&ChoiceValue::String("".to_string()), &constraints)); // too short
        assert!(!choice_permitted(&ChoiceValue::String("abcdef".to_string()), &constraints)); // too long
        assert!(!choice_permitted(&ChoiceValue::String("abcd".to_string()), &constraints)); // 'd' not allowed
    }

    #[test]
    fn test_choice_permitted_bytes() {
        
        let constraints = Constraints::Bytes(BytesConstraints {
            min_size: 2,
            max_size: 4,
        });
        
        assert!(choice_permitted(&ChoiceValue::Bytes(vec![1, 2]), &constraints));
        assert!(choice_permitted(&ChoiceValue::Bytes(vec![1, 2, 3, 4]), &constraints));
        assert!(!choice_permitted(&ChoiceValue::Bytes(vec![1]), &constraints)); // too short
        assert!(!choice_permitted(&ChoiceValue::Bytes(vec![1, 2, 3, 4, 5]), &constraints)); // too long
    }
}