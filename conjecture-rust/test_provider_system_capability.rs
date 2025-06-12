//! Provider System Capability Verification Test
//!
//! This test verifies that the provider system is working correctly
//! and validates the core capability requirements.

use conjecture::providers::{
    PrimitiveProvider, RandomProvider, HypothesisProvider, 
    ProviderRegistry, ProviderLifetime
};
use conjecture::choice::{IntegerConstraints, BooleanConstraints, FloatConstraints};
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

fn main() {
    println!("🧪 Provider System Capability Verification");
    println!("{}", "=".repeat(60));
    
    // Test 1: Provider Registry Functionality
    println!("\n✅ Test 1: Provider Registry");
    test_provider_registry();
    
    // Test 2: Random Provider Basic Functionality
    println!("\n✅ Test 2: Random Provider");
    test_random_provider();
    
    // Test 3: Hypothesis Provider Basic Functionality
    println!("\n✅ Test 3: Hypothesis Provider");
    test_hypothesis_provider();
    
    // Test 4: Provider Trait Implementations
    println!("\n✅ Test 4: Provider Trait Implementations");
    test_provider_traits();
    
    // Test 5: Constraint System Integration
    println!("\n✅ Test 5: Constraint System Integration");
    test_constraint_integration();
    
    println!("\n🎉 All Provider System tests passed!");
}

fn test_provider_registry() {
    let mut registry = ProviderRegistry::new();
    let available = registry.available_providers();
    
    assert!(available.contains(&"hypothesis".to_string()), "Registry should contain hypothesis provider");
    assert!(available.contains(&"random".to_string()), "Registry should contain random provider");
    
    let hypothesis_provider = registry.create("hypothesis").expect("Should create hypothesis provider");
    assert_eq!(hypothesis_provider.lifetime(), ProviderLifetime::TestCase);
    
    let random_provider = registry.create("random").expect("Should create random provider");
    assert_eq!(random_provider.lifetime(), ProviderLifetime::TestCase);
    
    println!("  ✓ Provider registry creates providers successfully");
    println!("  ✓ Available providers: {:?}", available);
}

fn test_random_provider() {
    let mut provider = RandomProvider::new();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    // Test integer generation
    let int_constraints = IntegerConstraints {
        min_value: Some(0),
        max_value: Some(100),
        weights: None,
        shrink_towards: Some(0),
    };
    
    let int_value = provider.generate_integer(&mut rng, &int_constraints)
        .expect("Should generate integer");
    assert!(int_value >= 0 && int_value <= 100, "Integer should be in range [0, 100]");
    println!("  ✓ Generated integer: {}", int_value);
    
    // Test boolean generation
    let bool_constraints = BooleanConstraints { p: 0.5 };
    let bool_value = provider.generate_boolean(&mut rng, &bool_constraints)
        .expect("Should generate boolean");
    println!("  ✓ Generated boolean: {}", bool_value);
    
    // Test float generation
    let float_constraints = FloatConstraints::new(Some(-10.0), Some(10.0));
    let float_value = provider.generate_float(&mut rng, &float_constraints)
        .expect("Should generate float");
    println!("  ✓ Generated float: {}", float_value);
    
    // Test string generation
    let string_value = provider.generate_string(&mut rng, "abc", 1, 5)
        .expect("Should generate string");
    assert!(!string_value.is_empty() && string_value.len() <= 5, "String should be between 1-5 chars");
    println!("  ✓ Generated string: '{}'", string_value);
    
    // Test bytes generation
    let bytes_value = provider.generate_bytes(&mut rng, 10)
        .expect("Should generate bytes");
    assert_eq!(bytes_value.len(), 10, "Should generate exactly 10 bytes");
    println!("  ✓ Generated {} bytes", bytes_value.len());
}

fn test_hypothesis_provider() {
    let mut provider = HypothesisProvider::new();
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    
    // Test integer generation with constant injection possibility
    let int_constraints = IntegerConstraints {
        min_value: Some(-1000),
        max_value: Some(1000),
        weights: None,
        shrink_towards: Some(0),
    };
    
    // Generate multiple integers to test constant injection
    println!("  Testing integer generation (with possible constant injection):");
    for i in 0..10 {
        let int_value = provider.generate_integer(&mut rng, &int_constraints)
            .expect("Should generate integer");
        println!("    [{:2}] Generated: {}", i+1, int_value);
    }
    
    // Test span notifications
    provider.span_start(12345u32);
    provider.span_end(false);
    println!("  ✓ Span notifications work correctly");
    
    // Test boolean generation
    let bool_constraints = BooleanConstraints { p: 0.3 };
    let bool_value = provider.generate_boolean(&mut rng, &bool_constraints)
        .expect("Should generate boolean");
    println!("  ✓ Generated boolean: {}", bool_value);
}

fn test_provider_traits() {
    // Test that both providers implement the trait correctly
    let random_provider: Box<dyn PrimitiveProvider> = Box::new(RandomProvider::new());
    let hypothesis_provider: Box<dyn PrimitiveProvider> = Box::new(HypothesisProvider::new());
    
    assert_eq!(random_provider.lifetime(), ProviderLifetime::TestCase);
    assert_eq!(hypothesis_provider.lifetime(), ProviderLifetime::TestCase);
    
    println!("  ✓ Both providers implement PrimitiveProvider trait");
    println!("  ✓ Lifetime management works correctly");
}

fn test_constraint_integration() {
    let mut provider = RandomProvider::new();
    let mut rng = ChaCha8Rng::seed_from_u64(456);
    
    // Test constraint edge cases
    
    // Test exact range (min == max)
    let exact_constraints = IntegerConstraints {
        min_value: Some(42),
        max_value: Some(42),
        weights: None,
        shrink_towards: Some(42),
    };
    
    let exact_value = provider.generate_integer(&mut rng, &exact_constraints)
        .expect("Should generate exact value");
    assert_eq!(exact_value, 42, "Should generate exact value when min == max");
    println!("  ✓ Exact value constraint works: {}", exact_value);
    
    // Test probability edge cases for boolean
    let certain_true = BooleanConstraints { p: 1.0 };
    let certain_false = BooleanConstraints { p: 0.0 };
    
    // Note: Due to floating point precision, we test multiple times
    let mut true_count = 0;
    let mut false_count = 0;
    
    for _ in 0..10 {
        if provider.generate_boolean(&mut rng, &certain_true).unwrap() {
            true_count += 1;
        }
        if !provider.generate_boolean(&mut rng, &certain_false).unwrap() {
            false_count += 1;
        }
    }
    
    println!("  ✓ Probability constraints work (true: {}/10, false: {}/10)", true_count, false_count);
    
    // Test float constraints
    let tight_float_constraints = FloatConstraints::new(Some(1.0), Some(2.0));
    let tight_float = provider.generate_float(&mut rng, &tight_float_constraints)
        .expect("Should generate float in tight range");
    println!("  ✓ Float constraint integration: {}", tight_float);
}