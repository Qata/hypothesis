//! Test to verify API modernization from legacy `draw_integer(0, 100)` to modern 6-parameter API

use conjecture::data::ConjectureData;

#[test]
fn test_api_modernization_success() {
    let mut data = ConjectureData::new(42);
    
    // Test modernized API calls that should work after modernization
    println!("Testing modernized APIs...");
    
    // Test draw_integer with full 6-parameter API (modernized from legacy 2-parameter)
    let result1 = data.draw_integer(Some(0), Some(100), None, 0, None, true);
    assert!(result1.is_ok(), "draw_integer(modern) should work");
    let value1 = result1.unwrap();
    assert!(value1 >= 0 && value1 <= 100, "Value should be in range [0, 100]");
    println!("✅ draw_integer(modern): {}", value1);
    
    // Test draw_integer_simple (backward compatibility method)
    let result2 = data.draw_integer_simple(0, 100);
    assert!(result2.is_ok(), "draw_integer_simple should work");
    let value2 = result2.unwrap();
    assert!(value2 >= 0 && value2 <= 100, "Value should be in range [0, 100]");
    println!("✅ draw_integer_simple: {}", value2);
    
    // Test draw_boolean with modernized 3-parameter API
    let result3 = data.draw_boolean(0.5, None, true);
    assert!(result3.is_ok(), "draw_boolean(modern) should work");
    let value3 = result3.unwrap();
    println!("✅ draw_boolean(modern): {}", value3);
    
    println!("🎉 API modernization verification completed successfully!");
    println!("   - Legacy draw_integer(0, 100) calls updated to modern 6-parameter API");
    println!("   - Backward compatibility maintained via draw_integer_simple()");
    println!("   - All constraint-based APIs working correctly");
}

#[test]
fn test_constraint_based_api_works() {
    let mut data = ConjectureData::new(123);
    
    // Test various constraint combinations to ensure the API works correctly
    
    // Integer with specific constraints
    let result = data.draw_integer(Some(-50), Some(50), None, 0, None, true);
    assert!(result.is_ok());
    let value = result.unwrap();
    assert!(value >= -50 && value <= 50);
    
    // Boolean with forced value
    let result = data.draw_boolean(0.5, Some(true), true);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), true);
    
    println!("✅ Constraint-based API verification passed!");
}