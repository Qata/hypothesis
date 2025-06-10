# Conjecture Rust 2: Electric Boogaloo

> **📝 LIVING DOCUMENT**: This file serves as both project specification and development journal. Update it continuously with progress, discoveries, design decisions, and any new information. Treat it as your second brain and checklist for this rewrite project.

## Key Takeaways from Swift-Hypothesis Branch Analysis

### Critical Dependency Insight
- **Crypto Dependencies**: The swift-hypothesis branch replaced `crypto-hash` with pure Rust `sha1 = "0.10"` crate
- **Reason**: Eliminates OpenSSL dependencies that cause cross-compilation issues for Apple platforms (tvOS, watchOS, visionOS)
- **Impact**: Pure Rust crypto enables reliable cross-compilation without external system dependencies
- **Implementation**: Simple replacement in database.rs:
  ```rust
  // OLD: crypto_hash::{hex_digest, Algorithm}
  // NEW: sha1::{Sha1, Digest}
  use sha1::{Sha1, Digest};
  
  fn sha1_hex_digest(data: &[u8]) -> String {
      let mut hasher = Sha1::new();
      hasher.update(data);
      format!("{:x}", hasher.finalize())
  }
  ```

### Architecture Observations
- **Draw Tracking**: Explicit `start_draw()/stop_draw()` calls are critical for proper shrinking context
- **Caching Strategy**: Engine uses execution caching (10k limit) to prevent redundant test execution during shrinking
- **Shrinking Debug**: Extensive debug printing throughout shrinking process for visibility
- **Boolean Generation**: Uses 63-bit precision for boolean decisions to match Python's approach
- **Array Shrinking**: Sophisticated array length shrinking that identifies continuation patterns

### Project Mission

This is a complete rewrite of the Rust conjecture engine, designed to faithfully port Python Hypothesis's modern conjecture architecture to Rust. Unlike the previous `conjecture-rust` implementation which was based on an older design, this version will match Python's current sophisticated architecture as closely as possible.

## Core Architectural Differences

**Current conjecture-rust (byte-based):**
```rust
pub fn bits(&mut self, n_bits: u64) -> Result<u64, FailedDraw>
pub fn draw_boolean(&mut self, p: f64) -> Result<bool, FailedDraw>
```

**Target Python-style (choice-based):**
```python
def draw_integer(min_value=0, max_value=None, ...)
def draw_boolean(p=0.5, forced=None, ...)
def draw_bytes(size, ...)
```

The rewrite moves from Python's legacy byte-stream approach to the modern choice-aware system where every draw is a typed choice with constraints.

## Why This Rewrite is Needed

### Problems with Current Implementation
- **Outdated Architecture**: Based on Python's pre-2019 byte-stream design, missing modern improvements
- **Limited Shrinking**: Cannot leverage choice-aware shrinking for better minimization
- **Ruby Integration Issues**: FFI complications and performance bottlenecks
- **Cross-Platform Limitations**: OpenSSL dependencies prevent reliable cross-compilation
- **Maintenance Burden**: Divergent codebase makes it hard to keep up with Python improvements

### Benefits of New Architecture  
- **Better Shrinking**: Choice-aware shrinking produces smaller, more meaningful examples
- **Type Safety**: Rust's type system enforces correctness at compile time
- **Performance**: Native Rust performance without FFI overhead for core operations
- **Cross-Platform**: Pure Rust dependencies enable compilation to any target
- **Maintainability**: Architecture closely matches Python, making updates easier

## Success Criteria

### Phase 1 Complete When:
- [ ] ALL Python conjecture tests have been ported and are passing
- [ ] Additional comprehensive tests written for edge cases
- [ ] ChoiceNode can represent all Python choice types (Integer, Boolean, Bytes, Float)
- [ ] Constraints properly validate choice values
- [ ] Choice sequences can be recorded and replayed
- [ ] Test coverage exceeds 95% for all core choice functionality

### Project Complete When:
- [ ] Ruby test suite passes with new engine
- [ ] Performance matches or exceeds current implementation  
- [ ] Cross-compilation works for all target platforms
- [ ] Shrinking quality equals or beats Python Hypothesis
- [ ] Memory usage is reasonable (< 2x current implementation)

## Five-Phase Implementation Plan

### Phase 1: Core Choice System 🚧
- [ ] Port ALL Python conjecture tests (RED phase - tests should fail initially)
- [ ] Write additional comprehensive tests for edge cases
- [ ] Choice types and constraints (implement to pass tests)
- [ ] Choice node implementation (implement to pass tests)
- [ ] Basic choice validation (implement to pass tests)

### Phase 2: ConjectureRunner & Data
- [ ] TestData equivalent 
- [ ] ConjectureRunner engine
- [ ] Choice recording and replay

### Phase 3: Modern Shrinking
- [ ] Choice-aware shrinker
- [ ] Constraint-preserving minimization
- [ ] Duplicate detection and removal

### Phase 4: Ruby Integration  
- [ ] Ruby bindings via Rutie
- [ ] Strategy integration
- [ ] Error handling and panics

### Phase 5: Advanced Features
- [ ] Targeting and coverage
- [ ] Statistical tracking
- [ ] Performance optimization

## Target Architecture

```
src/
├── choice/
│   ├── mod.rs            # Choice types and core traits
│   ├── constraints.rs    # Constraint definitions
│   ├── node.rs          # ChoiceNode implementation  
│   └── values.rs        # ChoiceValue types
├── data/
│   ├── mod.rs           # TestData and ConjectureData
│   ├── buffer.rs        # Data buffer management
│   └── status.rs        # Test result status types
├── engine/
│   ├── mod.rs           # ConjectureRunner
│   ├── runner.rs        # Main test execution
│   └── phases.rs        # Test phases (generate, shrink, etc)
├── shrinking/
│   ├── mod.rs           # Shrinker trait and core logic
│   ├── passes/          # Individual shrinking passes
│   │   ├── adaptive.rs  # Adaptive deletion
│   │   ├── minimize.rs  # Value minimization  
│   │   └── reorder.rs   # Block reordering
│   └── choice_aware.rs  # Choice-aware shrinking
├── database/
│   ├── mod.rs           # Database abstraction
│   ├── directory.rs     # File-based storage
│   └── memory.rs        # In-memory storage
├── strategies/          # Ruby integration layer
│   ├── mod.rs           # Strategy trait definitions
│   ├── primitives.rs    # Basic strategy implementations
│   └── ruby/            # Ruby-specific bindings
│       ├── mod.rs       # Ruby integration
│       ├── bindings.rs  # FFI definitions
│       └── strategies.rs     # Ruby strategy helpers
```

This architecture closely mirrors Python's organization while taking advantage of Rust's strengths in type safety and performance.

## Reference Materials

### Key Python Source Files
- **Choice System**: `hypothesis/internal/conjecture/data.py` - TestData and choice recording
- **Engine**: `hypothesis/internal/conjecture/engine.py` - ConjectureRunner main loop  
- **Shrinking**: `hypothesis/internal/conjecture/shrinker.py` - Choice-aware shrinking
- **Choices**: `hypothesis/internal/conjecture/choices.py` - Choice types and constraints

### Documentation
- **Conjecture Architecture**: https://hypothesis.readthedocs.io/en/latest/internals.html
- **Ruby Integration**: Current `hypothesis-ruby/` implementation for FFI patterns
- **Swift Integration**: `hypothesis-swift/` for cross-platform build insights

## API Design Principles

### Test-Driven Development Workflow
1. **RED**: Write failing tests first (port from Python + write additional)
2. **GREEN**: Implement minimal code to make tests pass
3. **REFACTOR**: Improve code quality while keeping tests green
4. **REPEAT**: Cycle through RED-GREEN-REFACTOR for each feature

### Rust Idioms to Follow
- **Error Handling**: Use `Result<T, E>` for fallible operations, avoid panics in library code
- **Ownership**: Prefer owned data structures, use borrowing for read-only access
- **Iterator Chains**: Leverage Rust's iterator patterns for data processing
- **Type Safety**: Use newtypes and enums to prevent invalid states
- **Zero-Cost Abstractions**: Design APIs that compile to efficient machine code

### Performance Targets
- **Memory**: Choice sequences should use arena allocation for cache locality
- **Execution**: Core choice operations should be inline-able and branch-predictable
- **Shrinking**: Avoid redundant constraint validation during shrinking passes
- **FFI**: Minimize Ruby ↔ Rust transitions, batch operations when possible

## Risk Assessment & Mitigation

### High-Risk Areas
1. **Ruby FFI Complexity**
   - *Risk*: Rutie integration becomes unwieldy or unstable
   - *Mitigation*: Start with minimal FFI surface, expand incrementally
   
2. **Performance Regression**
   - *Risk*: New architecture is slower than current implementation
   - *Mitigation*: Early benchmarking, profile-guided optimization
   
3. **Choice System Complexity**
   - *Risk*: Over-engineering leads to unmaintainable code
   - *Mitigation*: Start simple, add complexity only when needed

4. **Shrinking Quality**
   - *Risk*: New shrinker produces worse examples than current implementation
   - *Mitigation*: Port proven algorithms first, optimize later

### Medium-Risk Areas
- **Cross-Platform Builds**: Test early and often on target platforms
- **Memory Usage**: Monitor allocation patterns, especially for large test runs
- **Dependency Management**: Keep dependency tree minimal and well-audited

---

## Development Journal

### Session Notes
*Keep track of progress, blockers, and insights as development proceeds*

#### 2025-01-06: Project Initialization
- **Created**: Project directory and initial CLAUDE.md specification
- **Analysis Complete**: Comprehensive analysis of Python's conjecture architecture
- **Key Insight**: Python's choice system is perfectly suited for Rust's type system
- **Next Steps**: Begin Phase 1 implementation of core choice system

#### 2025-01-06: Swift-Hypothesis Branch Analysis
- **Analyzed**: swift-hypothesis branch for crypto dependency changes
- **Key Finding**: Migration from crypto-hash to sha1 crate eliminates OpenSSL dependencies
- **Architecture Insights**: Draw tracking, caching strategies, and sophisticated shrinking observed
- **Decision**: Use sha1 = "0.10" in new implementation for cross-platform compatibility

#### Future Sessions:
*Update this section with each development session*

### Design Decisions Log
*Record key architectural and implementation decisions*

- **Choice Types**: Will use Rust enum with associated data for type safety
- **Constraints**: Separate struct types for each choice type's constraints
- **Memory Management**: Plan to use arena allocation for choice sequences
- **Error Handling**: Standard Rust Result types throughout
- **Crypto Dependencies**: Use pure Rust `sha1` crate instead of `crypto-hash` to avoid OpenSSL dependencies
- **Draw Tracking**: Maintain explicit start_draw/stop_draw calls for proper shrinking context

### Blockers & Solutions
*Track problems encountered and how they were resolved*

*None yet - update as development proceeds*

### Performance Notes
*Track performance considerations and optimizations*

- **Caching Strategy**: Current Rust implementation uses 10k execution cache limit during shrinking
- **Pure Rust Crypto**: sha1 crate provides better cross-compilation than crypto-hash with OpenSSL

### Testing Strategy
*Document testing approach and coverage*

**🚨 TDD IS PARAMOUNT**: Test-Driven Development is the core methodology for this project

**Phase 1 Testing Priority:**
1. **Port Python Tests**: Copy ALL relevant tests from `hypothesis-python/tests/conjecture/`
2. **Write Additional Tests**: Expand test coverage beyond Python's test suite
3. **Red-Green-Refactor**: Write failing tests first, then implement to make them pass
4. **Comprehensive Coverage**: Every choice type, constraint, and edge case must be tested

**Test Categories:**
- **Unit Tests**: Each choice type and constraint validation (COPY FROM PYTHON)
- **Integration Tests**: Choice sequences and replay functionality  
- **Parity Tests**: Compare results with Python Hypothesis on same inputs
- **Ruby Integration Tests**: FFI layer and strategy compatibility
- **Property Tests**: Use Hypothesis to test the implementation itself
- **Benchmark Tests**: Performance regression detection
- **Edge Case Tests**: Boundary conditions, overflow, underflow scenarios

## Development Environment

### Required Tools
- **Rust**: Latest stable (1.70+), with clippy and rustfmt
- **Ruby**: 3.0+ with bundler for testing Ruby integration
- **Python**: 3.8+ with Hypothesis for parity testing
- **Git**: For branch analysis and development workflow

### Recommended Setup  
```bash
# Install Rust with common components
rustup component add clippy rustfmt

# Ruby development (for testing integration)
bundle install  

# Python for parity verification
pip install hypothesis

# Development tools
cargo install cargo-watch cargo-expand
```

### Build Commands
```bash
# TDD Development cycle
cargo test           # Run all tests (RED/GREEN verification)
cargo check          # Fast syntax/type checking
cargo clippy         # Linting
cargo fmt            # Formatting

# Test-driven workflow
cargo test --lib                    # Unit tests only
cargo test test_choice_integer      # Specific test
cargo test -- --nocapture          # See println! output
cargo watch -x test                 # Auto-run tests on file changes

# Integration testing  
cargo test --test ruby_integration
cargo bench          # Performance benchmarks
```

## Scope & Non-Goals

### In Scope for This Rewrite
- ✅ **Core Choice System**: All Python choice types faithfully ported
- ✅ **Modern Shrinking**: Choice-aware shrinking algorithms
- ✅ **Ruby Integration**: Full FFI layer for hypothesis-ruby
- ✅ **Cross-Platform**: Support for all Rust compilation targets
- ✅ **Performance**: Match or exceed current Rust implementation

### Explicitly Out of Scope
- ❌ **Python Bindings**: This is Rust-native, not a Python extension
- ❌ **Database Migration**: Will require manual migration from old format
- ❌ **Backward Compatibility**: Clean break from old byte-stream API
- ❌ **Advanced Targeting**: Defer sophisticated targeting features to Phase 5
- ❌ **Custom Strategies**: Ruby-side strategy composition, not Rust-side

### Deferred to Future Versions
- **Advanced Optimizations**: SIMD, custom allocators, etc.
- **Additional Language Bindings**: C, Swift, etc.
- **Distributed Testing**: Multi-machine test execution
- **GUI Tools**: Visual debugging and example exploration

---

## Checklist Management

**Instructions for Updating:**
1. ✅ Mark completed items with checkmarks
2. 🚧 Mark in-progress items with construction emoji
3. ❌ Mark blocked items with X and note reason
4. ➡️ Update with detailed progress notes
5. 📝 Add new discoveries to relevant sections
6. 🔄 Revise plans based on new information

**Current Phase**: 🚧 Phase 1: Core Choice System