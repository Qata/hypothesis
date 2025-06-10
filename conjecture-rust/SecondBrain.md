# SecondBrain.md - Cross-Project Insights & Patterns

> **Living Document**: Cross-project insights, patterns, and reusable approaches from the Conjecture Rust 2 rewrite project. Updated continuously with discoveries that could benefit future projects.

## Test-Driven Development Patterns

### RED-GREEN-REFACTOR with Cross-Language Parity
**Pattern**: When porting from one language to another, write failing tests that exactly match the source language's behavior first.

**Implementation Strategy**:
1. **RED**: Port source language tests directly, expect them to fail
2. **GREEN**: Implement minimal code to make tests pass  
3. **REFACTOR**: Improve implementation while keeping tests green
4. **EXPAND**: Add comprehensive test cases beyond source coverage

**Key Insight**: Source language test files contain the exact expected behavior including edge cases that may not be obvious from reading the implementation.

### Comprehensive Test Organization
**Pattern**: Organize tests into clear categories for systematic coverage:

- **Unit Tests**: Individual component behavior
- **Integration Tests**: Component interaction  
- **Parity Tests**: Direct ports ensuring cross-language compatibility
- **Property Tests**: Use property-based testing to verify the implementation
- **Performance Tests**: Regression detection and optimization targets
- **Edge Case Tests**: Boundary conditions and error scenarios

**Quality Standard**: Target 95%+ test coverage with meaningful test cases.

### FFI Verification Architecture
**Pattern**: Create separate verification crates to avoid polluting main library with cross-language validation code.

**Implementation**:
```rust
// verification-tests/
//   ├── src/
//   │   ├── main.rs           // CLI tool
//   │   ├── ffi_interface.rs  // Foreign function interface
//   │   ├── test_cases.rs     // Test case definitions
//   │   └── test_runner.rs    // Test execution engine
//   └── Cargo.toml            // Separate dependency management
```

**Benefits**: Clean separation, dedicated dependencies (like PyO3), independent build/test cycles.

## Cross-Language FFI Patterns

### Modern PyO3 API Usage
**Pattern**: Use the latest PyO3 API patterns for robust Python-Rust integration.

**Key APIs**:
```rust
use pyo3::prelude::*;
use pyo3::conversion::IntoPyObjectExt;

// Convert Rust → Python
let py_value = rust_value.into_bound_py_any(py)?;

// Convert Python → Rust  
let rust_value: RustType = py_value.extract()?;

// Call Python functions
let result = python_module.getattr("function_name")?.call1((args,))?;
```

**Critical**: Always import `IntoPyObjectExt` trait to access conversion methods.

### Subprocess vs FFI Decision Matrix
**Use FFI when**: High performance, frequent calls, complex data structures
**Use subprocess when**: Simple validation, one-off verification, debugging

**FFI Advantages**: Performance, type safety, integration
**Subprocess Advantages**: Isolation, debugging, simpler error handling

## Rust Implementation Patterns

### Type-Safe Choice Systems
**Pattern**: Use Rust enums with associated data for variant handling.

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ChoiceValue {
    Integer(i128),
    Boolean(bool), 
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constraints {
    Integer(IntegerConstraints),
    Boolean(BooleanConstraints),
    Float(FloatConstraints),
    // ...
}
```

**Benefit**: Compile-time correctness and exhaustive pattern matching.

### Mathematical Algorithm Implementation
**Pattern**: For complex mathematical operations, implement dual approaches:

1. **Formula-based**: For unbounded/infinite ranges
2. **Enumeration-based**: For bounded/finite ranges

**Example**: Choice indexing uses formulas for unbounded integers but sequence generation for bounded ranges to ensure exact ordering compatibility.

### Cross-Platform Dependency Management
**Pattern**: Prefer pure Rust implementations over system dependencies.

**Specific Example**: Use `sha1 = "0.10"` instead of `crypto-hash` to avoid OpenSSL dependencies and enable cross-compilation to all targets.

## Debugging and Development Patterns

### Extensive Debug Output Strategy
**Pattern**: Add comprehensive debug printing during complex algorithm development.

```rust
println!("ALGORITHM DEBUG: Converting {} with constraints {:?}", value, constraints);
println!("STEP DEBUG: Intermediate result = {}", intermediate);
```

**Benefits**: 
- Real-time algorithm behavior visibility
- Step-by-step verification against reference implementations
- Essential for mathematical algorithm debugging

**Production**: Use conditional compilation or feature flags to exclude from release builds.

## Quality Assurance Patterns

### Perfect Parity Verification
**Pattern**: Achieve and verify exact behavioral equivalence between implementations.

**Verification Strategy**:
1. **Direct Comparison**: Call both implementations with identical inputs
2. **Roundtrip Testing**: Verify inverse operations preserve data
3. **Property Testing**: Use the source implementation to test the target
4. **Edge Case Validation**: Test boundary conditions and error cases

**Success Metrics**: 
- 100% test pass rate on representative test suites
- Perfect roundtrip property for all valid inputs
- Identical behavior on edge cases and error conditions

### Test Suite Expansion Strategy
**Pattern**: Start with source language tests, then expand coverage systematically.

**Expansion Areas**:
- Boundary conditions beyond source tests
- Performance stress testing  
- Memory usage validation
- Error condition handling
- Integration scenarios

## Architecture Evolution Insights

### Legacy Migration Strategy
**Pattern**: When migrating from legacy architecture, target the modern destination rather than maintaining backward compatibility.

**Approach**:
1. Study current state-of-the-art implementation thoroughly
2. Ignore legacy constraints and technical debt
3. Build for future requirements, not legacy compatibility
4. Use extensive testing to ensure correctness

### Clean Codebase Principles
**Pattern**: Prioritize code clarity and maintainability over premature optimization.

**Guidelines**:
- Clear function and variable naming
- Comprehensive error handling
- Minimal external dependencies
- Extensive test coverage
- Living documentation

## Performance Optimization Patterns

### Memory-Efficient Collection Handling
**Pattern**: For operations that could generate large datasets, use lazy evaluation and bounds checking.

**Implementation**:
- Generate sequences on-demand rather than pre-computing
- Implement reasonable size limits to prevent memory exhaustion
- Use iterators for memory-efficient processing

### Benchmarking Strategy
**Pattern**: Establish performance baselines early and monitor for regressions.

**Targets**:
- Core operations < 1μs for typical inputs
- Memory usage comparable to or better than existing implementations
- Test suite execution time manageable for development workflow

## Project Management Patterns

### Living Documentation Strategy
**Pattern**: Maintain both project-specific and cross-project documentation.

**CLAUDE.md**: Project requirements, constraints, implementation details
**SecondBrain.md**: Reusable patterns, architectural insights, cross-project learnings
**DevJournal.md**: Chronological development log with decisions and context

### Structured Progress Tracking
**Pattern**: Use structured todo lists with clear status and priority indicators.

**Format**: JSON-like structure with status tracking enables programmatic progress monitoring.

## Algorithm Complexity Handling

### Constraint-Aware Algorithm Selection
**Pattern**: Choose different algorithmic approaches based on constraint characteristics.

**Examples**:
- **Unbounded ranges**: Mathematical formulas
- **Bounded ranges**: Sequence enumeration  
- **Probability-based**: Direct mapping calculations

**Key Insight**: One-size-fits-all approaches often fail for constraint-based systems.

### Overflow-Safe Mathematical Operations
**Pattern**: Use checked arithmetic and early bounds validation for mathematical operations.

```rust
if let Some(result) = base.checked_pow(exponent) {
    result
} else {
    return Err("Mathematical overflow");
}
```

**Critical for**: Preventing crashes in mathematical sequence operations.

## Future Application Opportunities

### Reusable Patterns for Other Projects
1. **Cross-Language Parity**: Test-driven approach for language porting
2. **FFI Architecture**: Clean separation between core library and language bindings
3. **Mathematical Algorithm Implementation**: Dual approach patterns
4. **Constraint-Based Systems**: Type-safe variant handling with Rust enums
5. **Cross-Platform Development**: Pure language implementation preferences

### Scaling Considerations
- **Memory**: Monitor allocation patterns for production usage
- **Performance**: Profile critical paths for optimization opportunities  
- **Concurrency**: Consider parallel execution for independent operations
- **Distribution**: Cross-compilation testing for all target platforms

## Current Implementation Status & Checklist

### Analysis Phase ✅ COMPLETED
- ☒ Systematically catalog every function in conjecture/data.py
- ☒ Update MISSING_FUNCTIONALITY.md with data.py function analysis
- ☒ Systematically catalog every function in conjecture/engine.py
- ☒ Update MISSING_FUNCTIONALITY.md with engine.py function analysis
- ☒ Systematically catalog every function in conjecture/choice.py
- ☒ Update MISSING_FUNCTIONALITY.md with choice.py function analysis
- ☒ Systematically catalog every function in conjecture/shrinker.py
- ☒ Update MISSING_FUNCTIONALITY.md with shrinker.py function analysis
- ☒ Systematically catalog every function in conjecture/providers.py
- ☒ Update MISSING_FUNCTIONALITY.md with providers.py function analysis
- ☒ Systematically catalog every function in conjecture/datatree.py
- ☒ Update MISSING_FUNCTIONALITY.md with datatree.py function analysis

**Key Discovery**: Our Rust implementation covers only ~15% of Python Hypothesis functionality. Missing critical systems include Span tracking, ConjectureRunner, DataTree, Provider system, and advanced shrinking infrastructure.

### Implementation Phase 🚧 IN PROGRESS

#### Phase 1: Core Choice System Extensions
- ☒ FloatConstraints implementation with required fields
- ☒ Python-compatible collection indexing functions (`python_size_to_index`, `python_index_to_size`, `collection_index`, `collection_value`)
- ☐ ChoiceNode.trivial property implementation
- ☐ Float choice indexing integration
- ☐ String/Bytes choice indexing support
- ☐ Extended choice_permitted validation

#### Phase 2: Infrastructure Components
- ☐ Status enum (INVALID/OVERRUN/INTERESTING states)
- ☐ Basic Span system for hierarchical choice tracking
- ☐ ConjectureResult with complete field set
- ☐ Observer pattern for ConjectureData
- ☐ Provider trait and HypothesisProvider basics

#### Phase 3: Engine & Execution
- ☐ ConjectureRunner for test execution
- ☐ DataTree infrastructure for novel prefix generation
- ☐ Constant-aware generation (edge case injection)

### Current Working Status

**Last Completed**: Python-compatible collection indexing functions
- Successfully implemented exact Python algorithms for collection ordering
- All roundtrip tests passing with perfect parity
- Functions ready for string/bytes choice integration

**Last Completed**: ChoiceNode.trivial property implementation
- Successfully implemented Python's exact trivial logic
- Handles forced nodes, non-float types (via choice_from_index(0)), and complex float cases
- All tests passing with perfect Python algorithm parity

**Next Priority**: Float choice indexing integration
- Extend choice_to_index/choice_from_index to support float types
- Already has basic implementation but needs integration with constraints
- Critical for complete choice system functionality

### Implementation Insights Discovered

#### Collection Indexing Algorithm Complexity
**Challenge**: Python's collection indexing uses sophisticated mathematical algorithms that must be implemented exactly for compatibility.

**Solution Pattern**: 
1. Implement exact Python algorithms with identical mathematical formulas
2. Handle edge cases like floating-point precision errors with integer-only fallbacks
3. Use comprehensive roundtrip testing to verify correctness

**Key Functions Implemented**:
- `python_size_to_index`: Geometric series formula for size-to-cumulative-index mapping
- `python_index_to_size`: Inverse logarithmic calculation with precision fallback
- `collection_index`: Element-wise ordering for collections (strings, bytes)
- `collection_value`: Index-to-collection conversion with bounds checking

#### Constraint System Evolution
**Discovery**: Python's FloatConstraints uses required `smallest_nonzero_magnitude` field, not optional.

**Impact**: Required updating all existing code that assumed Optional field.

**Pattern**: When porting constraint systems, verify exact field requirements rather than assuming optional fields.

#### Testing Strategy Effectiveness
**Pattern Validation**: TDD approach with extensive debug output proved essential for mathematical algorithm debugging.

**Success Metrics**: 
- Collection indexing: 100% roundtrip accuracy
- Size indexing: Perfect mathematical parity with Python
- Edge cases: Proper handling of boundary conditions

### Architecture Evolution Notes

#### Missing Functionality Gap Analysis
**Scale**: 85% of Python functionality missing from current implementation
**Priority Order**:
1. **DataTree** (40-50% of core functionality) - Novel prefix generation
2. **ConjectureRunner** (25% of functionality) - Test execution engine  
3. **Provider System** (15% of functionality) - Advanced generation algorithms
4. **Span System** (10% of functionality) - Hierarchical choice tracking

#### Implementation Approach Refinement
**Lesson**: Start with exact Python algorithm ports rather than creating "improved" versions.
**Reasoning**: Compatibility is more important than optimization in initial implementation.

**Future Optimization Strategy**: 
1. Achieve perfect parity first
2. Profile for performance bottlenecks
3. Optimize while maintaining compatibility

---

*This document captures key insights and reusable patterns from the Conjecture Rust 2 rewrite project. Updated continuously with discoveries from implementation and verification work.*