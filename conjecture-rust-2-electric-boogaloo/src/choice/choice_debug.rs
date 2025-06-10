//! Debug helper to understand Python's choice indexing algorithm

use super::*;

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::choice::*;

    #[test]
    fn analyze_python_ordering() {
        println!("CHOICE_DEBUG: Analyzing Python's integer choice ordering");
        
        // Test case: integer_constr(-3, 3, shrink_towards=1), (1, 2, 0, 3, -1, -2, -3)
        let expected_order = vec![1i128, 2, 0, 3, -1, -2, -3];
        let shrink_towards = 1i128;
        
        println!("CHOICE_DEBUG: Expected order: {:?}", expected_order);
        println!("CHOICE_DEBUG: shrink_towards: {}", shrink_towards);
        
        for (index, value) in expected_order.iter().enumerate() {
            let distance_from_shrink = (*value - shrink_towards).abs();
            let is_positive = *value > shrink_towards;
            let is_negative = *value < shrink_towards;
            
            println!("CHOICE_DEBUG: Index {}: value={}, distance={}, positive={}, negative={}", 
                index, value, distance_from_shrink, is_positive, is_negative);
        }
        
        // Analyze the pattern:
        println!("CHOICE_DEBUG: Pattern analysis:");
        println!("CHOICE_DEBUG: Index 0: 1 (shrink_towards, distance=0)");
        println!("CHOICE_DEBUG: Index 1: 2 (distance=1, positive)");
        println!("CHOICE_DEBUG: Index 2: 0 (distance=1, negative)"); 
        println!("CHOICE_DEBUG: Index 3: 3 (distance=2, positive)");
        println!("CHOICE_DEBUG: Index 4: -1 (distance=2, negative)");
        println!("CHOICE_DEBUG: Index 5: -2 (distance=3, negative)");
        println!("CHOICE_DEBUG: Index 6: -3 (distance=4, negative)");
        
        // The pattern is: start with shrink_towards, then alternate positive/negative distances
        // BUT we need to be careful about bounds!
        
        // For distance 1: +1 (value 2), then -1 (value 0)
        // For distance 2: +2 (value 3), then -2 (value -1) 
        // For distance 3: +3 would be 4 (out of bounds), so skip to -3 (value -2)
        // For distance 4: +4 would be 5 (out of bounds), so skip to -4 (value -3)
        
        println!("CHOICE_DEBUG: The algorithm alternates +/- but skips out-of-bounds values!");
    }
}