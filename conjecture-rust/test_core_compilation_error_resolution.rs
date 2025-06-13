//! Test the Core Compilation Error Resolution module
//! This is a standalone test to verify the core functionality works correctly

use conjecture::choice::{
    CompilationErrorType, CompilationErrorResolver, ResolutionResult, 
    ChoiceNodeBuilder, CompilationErrorAnalyzer, ChoiceType, ChoiceValue, 
    Constraints, IntegerConstraints
};

fn main() {
    println!("Testing Core Compilation Error Resolution implementation...");
    
    // Test 1: Basic resolver creation
    println!("\n1. Testing resolver creation...");
    let mut resolver = CompilationErrorResolver::new();
    let stats = resolver.get_statistics();
    assert_eq!(stats.total_errors_analyzed, 0);
    println!("✓ Resolver created successfully");
    
    // Test 2: Import path error resolution
    println!("\n2. Testing import path error resolution...");
    let import_error = CompilationErrorType::ImportPathError {
        invalid_path: "conjecture_rust".to_string(),
        suggested_path: "conjecture".to_string(),
    };
    
    let result = resolver.resolve_error(import_error);
    match result {
        ResolutionResult::Resolved { confidence, fix_applied, .. } => {
            assert!(confidence > 0.9);
            assert!(fix_applied.contains("conjecture_rust"));
            assert!(fix_applied.contains("conjecture"));
            println!("✓ Import path error resolved with confidence: {:.2}%", confidence * 100.0);
            println!("  Fix: {}", fix_applied);
        }
        _ => panic!("Expected resolved result for import path error"),
    }
    
    // Test 3: Missing trait implementation error
    println!("\n3. Testing trait implementation error resolution...");
    let trait_error = CompilationErrorType::MissingTraitImplementation {
        trait_name: "Clone".to_string(),
        target_type: "TestStruct".to_string(),
    };
    
    let result = resolver.resolve_error(trait_error);
    match result {
        ResolutionResult::Resolved { confidence, fix_applied, .. } => {
            assert!(confidence > 0.8);
            assert!(fix_applied.contains("Clone"));
            println!("✓ Trait implementation error resolved with confidence: {:.2}%", confidence * 100.0);
            println!("  Fix: {}", fix_applied);
        }
        _ => panic!("Expected resolved result for trait error"),
    }
    
    // Test 4: Field access error resolution
    println!("\n4. Testing field access error resolution...");
    let field_error = CompilationErrorType::FieldAccessError {
        struct_name: "ChoiceNode".to_string(),
        field_name: "index".to_string(),
        available_fields: vec!["choice_type".to_string(), "value".to_string()],
    };
    
    let result = resolver.resolve_error(field_error);
    match result {
        ResolutionResult::Resolved { confidence, fix_applied, .. } => {
            assert!(confidence > 0.8);
            assert!(fix_applied.contains("with_index"));
            println!("✓ Field access error resolved with confidence: {:.2}%", confidence * 100.0);
            println!("  Fix: {}", fix_applied);
        }
        _ => panic!("Expected resolved result for field access error"),
    }
    
    // Test 5: ChoiceNodeBuilder functionality
    println!("\n5. Testing ChoiceNodeBuilder...");
    let node = ChoiceNodeBuilder::new()
        .choice_type(ChoiceType::Integer)
        .value(ChoiceValue::Integer(42))
        .constraints(Constraints::Integer(IntegerConstraints::default()))
        .was_forced(false)
        .index(0)
        .build()
        .unwrap();
    
    assert_eq!(node.choice_type, ChoiceType::Integer);
    assert_eq!(node.value, ChoiceValue::Integer(42));
    assert!(!node.was_forced);
    assert_eq!(node.index, Some(0));
    println!("✓ ChoiceNodeBuilder created node successfully");
    
    // Test 6: Error analyzer
    println!("\n6. Testing compilation error analyzer...");
    let mut analyzer = CompilationErrorAnalyzer::new();
    
    let error_msg = "failed to resolve: use of unresolved module or unlinked crate `conjecture_rust`";
    let results = analyzer.analyze_and_resolve(error_msg);
    
    assert!(!results.is_empty());
    println!("✓ Error analyzer identified {} potential fixes", results.len());
    
    for (i, result) in results.iter().enumerate() {
        match result {
            ResolutionResult::Resolved { original_error, fix_applied, confidence } => {
                println!("  Fix {}: {} (confidence: {:.1}%)", i + 1, fix_applied, confidence * 100.0);
            }
            ResolutionResult::RequiresManualFix { suggestions, .. } => {
                println!("  Manual fix required. Suggestions: {:?}", suggestions);
            }
            ResolutionResult::Unresolvable { reason, .. } => {
                println!("  Unresolvable: {}", reason);
            }
        }
    }
    
    // Test 7: Statistics verification
    println!("\n7. Testing resolution statistics...");
    let final_stats = resolver.get_statistics();
    assert!(final_stats.total_errors_analyzed >= 3); // We resolved at least 3 errors
    assert!(final_stats.successful_resolutions >= 3);
    assert!(final_stats.resolution_confidence_average > 0.8);
    
    println!("✓ Statistics: {} errors analyzed, {} resolved successfully", 
             final_stats.total_errors_analyzed, final_stats.successful_resolutions);
    println!("  Average confidence: {:.1}%", final_stats.resolution_confidence_average * 100.0);
    
    // Test 8: Report generation
    println!("\n8. Testing report generation...");
    let report = resolver.generate_resolution_report();
    assert!(report.contains("COMPILATION ERROR RESOLUTION REPORT"));
    assert!(report.contains("Resolution Rate"));
    println!("✓ Resolution report generated successfully");
    
    let analysis_report = analyzer.generate_analysis_report();
    assert!(analysis_report.contains("ERROR PATTERN ANALYSIS"));
    println!("✓ Analysis report generated successfully");
    
    println!("\n🎉 All tests passed! Core Compilation Error Resolution implementation is working correctly.");
    println!("\nKey capabilities implemented:");
    println!("• Automatic import path correction (conjecture_rust → conjecture)");
    println!("• Missing trait implementation resolution");
    println!("• Struct field access error resolution");
    println!("• ChoiceNodeBuilder for safe construction");
    println!("• Comprehensive error analysis and reporting");
    println!("• Statistical tracking with confidence metrics");
}