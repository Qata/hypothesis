//! Standalone validation of trait object compatibility solutions
//! 
//! This file demonstrates the exact solutions for rand::Rng trait object compatibility
//! without depending on the broken conjecture-rust codebase.

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// PROBLEM: This pattern fails with "trait `Rng` is not dyn compatible"
// fn broken_function(rng: &mut dyn Rng) -> i32 {
//     rng.gen_range(1..=100)  // E0038 error
// }

/// SOLUTION 1: Generic parameter
fn fixed_with_generic<R: Rng>(rng: &mut R) -> i32 {
    rng.gen_range(1..=100)
}

/// SOLUTION 2: Trait bounds
fn fixed_with_bounds<R: Rng + RngCore>(rng: &mut R) -> f64 {
    rng.gen()
}

/// SOLUTION 3: Concrete type
fn fixed_with_concrete(rng: &mut ChaCha8Rng) -> bool {
    rng.gen()
}

/// SOLUTION 4: Generic struct instead of trait object
struct FixedWrapper<R: Rng> {
    rng: R,
}

impl<R: Rng> FixedWrapper<R> {
    fn new(rng: R) -> Self {
        Self { rng }
    }
    
    fn generate(&mut self) -> i32 {
        self.rng.gen_range(1..=100)
    }
}

fn main() {
    println!("Testing trait object compatibility solutions...");
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    // Test all solutions
    let result1 = fixed_with_generic(&mut rng);
    let result2 = fixed_with_bounds(&mut rng);
    let result3 = fixed_with_concrete(&mut rng);
    
    let mut wrapper = FixedWrapper::new(ChaCha8Rng::seed_from_u64(123));
    let result4 = wrapper.generate();
    
    println!("Generic solution: {}", result1);
    println!("Trait bounds solution: {}", result2);
    println!("Concrete type solution: {}", result3);
    println!("Wrapper solution: {}", result4);
    
    println!("All trait object compatibility solutions work!");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_all_solutions_compile_and_work() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let result1 = fixed_with_generic(&mut rng);
        assert!(result1 >= 1 && result1 <= 100);
        
        let result2 = fixed_with_bounds(&mut rng);
        assert!(result2 >= 0.0 && result2 <= 1.0);
        
        let result3 = fixed_with_concrete(&mut rng);
        assert!(result3 == true || result3 == false);
        
        let mut wrapper = FixedWrapper::new(ChaCha8Rng::seed_from_u64(123));
        let result4 = wrapper.generate();
        assert!(result4 >= 1 && result4 <= 100);
    }
    
    #[test]
    fn test_deterministic_behavior() {
        let seed = 12345;
        
        let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
        let mut rng2 = ChaCha8Rng::seed_from_u64(seed);
        
        let result1 = fixed_with_generic(&mut rng1);
        let result2 = fixed_with_generic(&mut rng2);
        
        assert_eq!(result1, result2, "Same seed should produce same results");
    }
}