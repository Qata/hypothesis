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

## Five-Phase Implementation Plan

### Phase 1: Core Choice System ✅
- [x] Choice types and constraints
- [x] Choice node implementation
- [x] Basic choice validation

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

*To be defined during Phase 1*

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