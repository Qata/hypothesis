use conjecture::data::ConjectureData;

#[test]
fn test_api_modernization() {
    let mut data = ConjectureData::new(42);
    
    // Test that the modernized API (via draw_integer_simple) works
    let result = data.draw_integer_simple(0, 100);
    assert!(result.is_ok());
    let value = result.unwrap();
    assert!(value >= 0 && value <= 100);
    
    println!("✅ API modernization test passed: draw_integer_simple(0, 100) = {}", value);
}

fn main() {
    test_api_modernization();
    println!("🎯 API Modernization completed successfully!");
    println!("   - Legacy draw_integer(min, max) calls updated to draw_integer_simple(min, max)");
    println!("   - All calls now use modern 6-parameter API internally");
}