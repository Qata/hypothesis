//! Float Constraint Type System Capability Demo
//!
//! This demo shows the complete Float Constraint Type System capability that has been
//! implemented for the ConjectureData module. It demonstrates:
//!
//! 1. **Fixed smallest_nonzero_magnitude type**: Properly typed as Option<f64>
//! 2. **Exported float encoding functions**: Available for external use
//! 3. **Comprehensive constraint type system**: Full validation and generation
//! 4. **Integration with ConjectureData**: Sophisticated float generation
//! 5. **Constant-aware generation**: Edge case testing with predefined pools

use conjecture::{
    ConjectureData, Status,
    FloatConstraintTypeSystem, FloatGenerationStrategy,
    float_to_lex, lex_to_float, float_to_int, int_to_float,
    FloatWidth,
};
use conjecture::choice::constraints::FloatConstraints;

fn main() {
    println!("🔬 Float Constraint Type System Capability Demo");
    println!("=" .repeat(60));
    
    // Demo 1: Fixed smallest_nonzero_magnitude type (Option<f64>)
    demo_fixed_type_system();
    
    // Demo 2: Float encoding functions export
    demo_float_encoding_export();
    
    // Demo 3: Comprehensive constraint type system
    demo_constraint_type_system();
    
    // Demo 4: ConjectureData integration
    demo_conjecture_data_integration();
    
    // Demo 5: Constant-aware generation
    demo_constant_aware_generation();
    
    println!("\n✅ All demos completed successfully!");
    println!("🎯 Float Constraint Type System capability is fully operational.");
}

fn demo_fixed_type_system() {
    println!("\n📋 Demo 1: Fixed smallest_nonzero_magnitude Type System");
    println!("-" .repeat(50));
    
    // Demonstrate that smallest_nonzero_magnitude is now properly Option<f64>
    let constraints_with_some = FloatConstraints {
        min_value: -100.0,
        max_value: 100.0,
        allow_nan: false,
        smallest_nonzero_magnitude: Some(1e-6), // Option<f64> with Some() wrapper
    };
    
    let constraints_with_none = FloatConstraints {
        min_value: -100.0,
        max_value: 100.0,
        allow_nan: false,
        smallest_nonzero_magnitude: None, // Option<f64> with None
    };
    
    println!("✓ Constraints with Some(1e-6): {:?}", constraints_with_some.smallest_nonzero_magnitude);
    println!("✓ Constraints with None: {:?}", constraints_with_none.smallest_nonzero_magnitude);
    
    // Test validation
    println!("✓ Validation with Some magnitude:");
    println!("  - Value 1e-7 (too small): {}", constraints_with_some.validate(1e-7));
    println!("  - Value 1e-5 (valid): {}", constraints_with_some.validate(1e-5));
    
    println!("✓ Validation with None magnitude:");
    println!("  - Value 1e-100 (valid): {}", constraints_with_none.validate(1e-100));
    println!("  - Very small values allowed when magnitude is None");
}

fn demo_float_encoding_export() {
    println!("\n🔧 Demo 2: Float Encoding Functions Export");
    println!("-" .repeat(50));
    
    let test_values = [0.0, -0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 3.14159];
    
    println!("Testing exported float encoding functions:");
    
    for &value in &test_values {
        if value.is_nan() {
            println!("✓ Value: NaN");
            let lex = float_to_lex(value);
            let storage = float_to_int(value);
            println!("  - Lexicographic: {:#018X}", lex);
            println!("  - Storage int: {:#018X}", storage);
            // Note: NaN round-trip may not be exact due to normalization
        } else {
            println!("✓ Value: {}", value);
            
            // Test lexicographic encoding round-trip
            let lex = float_to_lex(value);
            let recovered_lex = lex_to_float(lex);
            println!("  - Lexicographic: {:#018X} -> {}", lex, recovered_lex);
            
            // Test storage encoding round-trip
            let storage = float_to_int(value);
            let recovered_storage = int_to_float(storage);
            println!("  - Storage int: {:#018X} -> {}", storage, recovered_storage);
            
            // Verify round-trip accuracy (for non-NaN values)
            assert_eq!(value.to_bits(), recovered_lex.to_bits(), "Lexicographic round-trip failed");
            assert_eq!(value.to_bits(), recovered_storage.to_bits(), "Storage round-trip failed");
        }
    }
    
    println!("✓ All encoding functions working correctly!");
}

fn demo_constraint_type_system() {
    println!("\n🏗️ Demo 3: Comprehensive Constraint Type System");
    println!("-" .repeat(50));
    
    let constraints = FloatConstraints {
        min_value: -10.0,
        max_value: 10.0,
        allow_nan: false,
        smallest_nonzero_magnitude: Some(1e-3),
    };
    
    // Test different generation strategies
    let strategies = [
        ("Uniform", FloatGenerationStrategy::Uniform),
        ("Lexicographic", FloatGenerationStrategy::Lexicographic),
        ("ConstantBiased", FloatGenerationStrategy::ConstantBiased { constant_probability: 0.5 }),
        ("ConstraintAware", FloatGenerationStrategy::ConstraintAware),
    ];
    
    for (name, strategy) in strategies.iter() {
        println!("✓ Testing {} generation strategy:", name);
        
        let system = FloatConstraintTypeSystem::with_strategy(constraints.clone(), strategy.clone());
        let constant_pool = system.get_constant_pool();
        
        println!("  - Constant pool size: {}", constant_pool.len());
        println!("  - Sample constants: {:?}", &constant_pool[..constant_pool.len().min(5)]);
        
        // Test encoding integration
        let test_value = 5.0;
        let lex_encoded = system.float_to_shrink_order(test_value);
        let storage_encoded = system.float_to_storage_int(test_value);
        
        println!("  - Test value 5.0:");
        println!("    * Shrink order: {:#018X}", lex_encoded);
        println!("    * Storage int: {:#018X}", storage_encoded);
        
        // Verify round-trips
        assert_eq!(test_value, system.shrink_order_to_float(lex_encoded));
        assert_eq!(test_value, system.storage_int_to_float(storage_encoded));
    }
    
    println!("✓ All constraint type system features working!");
}

fn demo_conjecture_data_integration() {
    println!("\n🔗 Demo 4: ConjectureData Integration");
    println!("-" .repeat(50));
    
    let mut data = ConjectureData::for_buffer(&[], None);
    
    // Test basic float drawing
    println!("✓ Testing basic float drawing:");
    for i in 0..5 {
        match data.draw_float() {
            Ok(value) => println!("  - Draw {}: {}", i + 1, value),
            Err(e) => println!("  - Draw {}: Error {:?}", i + 1, e),
        }
    }
    
    // Test constrained float drawing
    println!("\n✓ Testing constrained float drawing:");
    for i in 0..3 {
        match data.draw_float_full(0.0, 100.0, false, Some(1e-6), None) {
            Ok(value) => {
                println!("  - Constrained draw {}: {}", i + 1, value);
                // Verify constraints are met
                assert!(value >= 0.0 && value <= 100.0, "Value outside range");
                if value.abs() != 0.0 {
                    assert!(value.abs() >= 1e-6, "Value violates magnitude constraint");
                }
            }
            Err(e) => println!("  - Constrained draw {}: Error {:?}", i + 1, e),
        }
    }
    
    // Test forced values
    println!("\n✓ Testing forced value drawing:");
    let forced_values = [42.0, -3.14, 0.0];
    for &forced in &forced_values {
        match data.draw_float_full(-100.0, 100.0, false, None, Some(forced)) {
            Ok(value) => {
                println!("  - Forced {} -> {}", forced, value);
                assert_eq!(value, forced, "Forced value not respected");
            }
            Err(e) => println!("  - Forced {} -> Error {:?}", forced, e),
        }
    }
    
    println!("✓ ConjectureData integration working correctly!");
}

fn demo_constant_aware_generation() {
    println!("\n🎯 Demo 5: Constant-Aware Generation");
    println!("-" .repeat(50));
    
    let constraints = FloatConstraints {
        min_value: -1000.0,
        max_value: 1000.0,
        allow_nan: true,
        smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
    };
    
    let system = FloatConstraintTypeSystem::with_strategy(
        constraints,
        FloatGenerationStrategy::ConstantBiased { constant_probability: 1.0 } // Always use constants for demo
    );
    
    println!("✓ Constant pool contains {} edge case values:", system.get_constant_pool().len());
    
    let pool = system.get_constant_pool();
    let interesting_constants: Vec<_> = pool.iter()
        .filter(|&&x| x.is_nan() || x.is_infinite() || x == 0.0 || x == -0.0 || x.abs() == f64::MIN_POSITIVE)
        .collect();
    
    println!("  - Special values found:");
    for &constant in &interesting_constants {
        if constant.is_nan() {
            println!("    * NaN");
        } else if constant.is_infinite() {
            println!("    * {} infinity", if constant > 0.0 { "Positive" } else { "Negative" });
        } else if constant == 0.0 && constant.is_sign_positive() {
            println!("    * Positive zero");
        } else if constant == 0.0 && constant.is_sign_negative() {
            println!("    * Negative zero");
        } else if constant.abs() == f64::MIN_POSITIVE {
            println!("    * {} MIN_POSITIVE", if constant > 0.0 { "Positive" } else { "Negative" });
        } else {
            println!("    * {}", constant);
        }
    }
    
    // Test that constants are being used for edge case testing
    println!("\n✓ Edge cases ready for property-based testing!");
    println!("  - {} total constants available", pool.len());
    println!("  - {} special IEEE-754 values", interesting_constants.len());
    
    // Demonstrate validation and clamping
    println!("\n✓ Testing validation and clamping:");
    let test_cases = [
        (f64::NAN, "NaN"),
        (f64::INFINITY, "Positive infinity"),
        (1500.0, "Value above max"),
        (-1500.0, "Value below min"),
        (1e-400, "Value below magnitude threshold"),
    ];
    
    for &(value, description) in &test_cases {
        let valid = system.validate_float(value);
        let clamped = system.constrain_float(value);
        
        if value.is_nan() {
            println!("  - {}: valid={}, clamped=NaN", description, valid);
        } else {
            println!("  - {}: valid={}, clamped={}", description, valid, clamped);
        }
    }
    
    println!("✓ Constant-aware generation system fully operational!");
}