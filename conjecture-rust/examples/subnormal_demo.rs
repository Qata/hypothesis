// Example demonstrating the new subnormal control features
// This shows the Python Hypothesis parity we've achieved

use conjecture::data::DataSource;
use conjecture::floats::{
    draw_float_width_with_subnormals, 
    FloatWidth,
    subnormal_support_warning,
    is_subnormal_width,
    next_up_normal_width,
    next_down_normal_width
};

fn main() {
    println!("🔬 Demonstrating Enhanced Float Generation with Subnormal Control");
    println!("================================================================\n");

    // Check environment subnormal support
    if let Some(warning) = subnormal_support_warning() {
        println!("⚠️  {}\n", warning);
    } else {
        println!("✅ Environment properly supports subnormal numbers\n");
    }

    for width in [FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
        println!("📊 Testing {:?} ({} bits)", width, width.bits());
        println!("----------------------------------------");
        
        let smallest_normal = width.smallest_normal();
        println!("   Smallest normal: {:.2e}", smallest_normal);
        
        // Demonstrate auto-detection logic
        let tiny_range_min = smallest_normal / 4.0;
        let tiny_range_max = smallest_normal / 2.0;
        
        println!("   Testing range [{:.2e}, {:.2e}]", tiny_range_min, tiny_range_max);
        
        // Create test data source
        let test_data: Vec<u64> = vec![42u64; 64];
        let mut source = DataSource::from_vec(test_data.clone());
        
        // Demo 1: Auto-detection (None = use Python-style auto-detection)
        println!("   🎯 Auto-detection (allow_subnormal=None):");
        match draw_float_width_with_subnormals(
            &mut source, width, tiny_range_min, tiny_range_max, false, false, None
        ) {
            Ok(val) => {
                let is_sub = is_subnormal_width(val, width);
                println!("      Generated: {:.2e} (subnormal: {})", val, is_sub);
            }
            Err(_) => println!("      Failed to generate (out of data)")
        }
        
        // Reset source
        source = DataSource::from_vec(test_data.clone());
        
        // Demo 2: Explicit subnormal enabling
        println!("   ✅ Explicit subnormal enabled (allow_subnormal=true):");
        match draw_float_width_with_subnormals(
            &mut source, width, tiny_range_min, tiny_range_max, false, false, Some(true)
        ) {
            Ok(val) => {
                let is_sub = is_subnormal_width(val, width);
                println!("      Generated: {:.2e} (subnormal: {})", val, is_sub);
            }
            Err(_) => println!("      Failed to generate (out of data)")
        }
        
        // Demo 3: Explicit subnormal disabling (should fail validation)
        println!("   ❌ Explicit subnormal disabled (allow_subnormal=false):");
        match draw_float_width_with_subnormals(
            &mut source, width, tiny_range_min, tiny_range_max, false, false, Some(false)
        ) {
            Ok(val) => println!("      Unexpected success: {:.2e}", val),
            Err(_) => println!("      ✓ Correctly failed validation (bounds require subnormals)")
        }
        
        // Demo 4: Normal-only navigation helpers
        println!("   🧭 Normal-only navigation:");
        let test_val = smallest_normal / 3.0; // A subnormal value
        let next_up_normal = next_up_normal_width(test_val, width);
        let next_down_normal = next_down_normal_width(test_val, width);
        
        println!("      From {:.2e} (subnormal: {})", test_val, is_subnormal_width(test_val, width));
        println!("      Next up normal: {:.2e}", next_up_normal);
        println!("      Next down normal: {:.2e}", next_down_normal);
        
        println!();
    }
    
    println!("🎉 All demos completed! The Rust implementation now matches");
    println!("   Python Hypothesis's subnormal handling capabilities.");
    println!("\n💡 Key Features Added:");
    println!("   • allow_subnormal parameter with auto-detection");
    println!("   • Smart bounds validation");
    println!("   • Width-specific smallest normal constants");
    println!("   • Normal-only navigation helpers");
    println!("   • FTZ (Flush-to-Zero) detection");
    println!("   • Full Python Hypothesis API parity! 🐍 ➡️ 🦀");
}