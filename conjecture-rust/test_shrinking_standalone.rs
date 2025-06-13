// Standalone test to demonstrate the shrinking implementation
use conjecture::choice::{ChoiceNode, ChoiceValue, Constraints, IntegerConstraints, ChoiceType};
use conjecture::data::ConjectureData;
use conjecture::shrinking::{shrink_integer, IntegerShrinker};

fn main() {
    // Test 1: Integer shrinking
    println!("Testing integer shrinking...");
    let predicate = |x: i128| x > 10;
    let result = shrink_integer(100, predicate);
    println!("Integer shrinking: 100 -> {} (should be 11)", result);
    assert!(result > 10 && result < 100);
    
    // Test 2: Manual integer shrinker
    println!("Testing manual integer shrinker...");
    let mut shrinker = IntegerShrinker::new(1000, Box::new(|x| x > 50));
    let shrunk = shrinker.shrink();
    println!("Manual integer shrinking: 1000 -> {} (should be 51)", shrunk);
    assert!(shrunk > 50 && shrunk < 1000);
    
    println!("All shrinking tests passed!");
}